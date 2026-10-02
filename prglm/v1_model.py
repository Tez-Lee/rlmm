"""PRG-v1: region-owned ternary edges and sparse recurrent message events.

The sampled/argmax engine visits only live walkers. The expected engine is a
soft region-mass surrogate: it retains every nonzero region branch, while node
locations are represented by one K-node pattern per region.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .v1_config import V1Config


def ternary_ste(x):
    quantized=(x > .33).to(x.dtype)-(x < -.33).to(x.dtype)
    return x+(quantized-x).detach()


def accumulator_step(old, incoming, enabled=True, decay=1.):
    return decay*old+incoming if enabled else incoming


def build_gateways(regions, gateways):
    # One small global-address table per region; no node owns a region address.
    rows=[]
    for region in range(regions):
        candidates=[(region+step) % regions for step in range(1,regions)]
        # Spread gateways around the graph instead of choosing only ring peers.
        order=sorted(candidates,key=lambda dest: ((dest-region)*17)%regions)
        rows.append(order[:gateways])
    return torch.tensor(rows,dtype=torch.long)


class PRGLMv1(nn.Module):
    def __init__(self,c: V1Config):
        super().__init__();self.c=c.validate()
        r,n,e,k,g=c.regions,c.region_size,c.edges_per_node,c.active_nodes,c.gateways
        self.emb=nn.Embedding(c.vocab_size,c.d_model)
        self.global_router=nn.Linear(c.d_model,r)
        self.seed_proj=nn.Linear(c.d_model,k)
        self.local_edge=nn.Parameter(torch.randn(r,n,e)*.65)
        self.region_emb=nn.Parameter(torch.randn(r,16)*.02)
        self.router=nn.Sequential(nn.Linear(22,32),nn.Tanh(),nn.Linear(32,g+3))
        self.transition_logits=nn.Parameter(torch.zeros(r,g+3))
        self.readout_proj=nn.Linear(k,c.d_model)
        self.norm=nn.LayerNorm(c.d_model)
        self.head=nn.Linear(c.d_model,c.vocab_size,bias=False)
        self.head.weight=self.emb.weight
        self.register_buffer('offsets',torch.tensor(list(range(-e//2,0))+list(range(1,e//2+1)),dtype=torch.long))
        self.register_buffer('gateway_table',build_gateways(r,g))

    def _initial_indices(self,token_ids,regions):
        k=self.c.active_nodes
        return (token_ids[...,None]+regions[...,None]*37+
                torch.arange(k,device=regions.device)*7)%self.c.region_size

    def _propagate(self,state,accum,b_idx,regions,indices,message,mass,accumulation=None):
        """Process only supplied live events; return K next node positions/event."""
        c=self.c;n,e=c.region_size,c.edges_per_node
        if b_idx.numel()==0:
            empty=indices.new_empty((0,c.active_nodes))
            return state*c.state_decay,accum*c.accumulator_decay,empty,empty.float(),empty.float(),0,0
        raw=self.local_edge[regions[:,None],indices]
        edge=ternary_ste(raw) if c.edge_type=='ternary' else raw
        dest=(indices[...,None]+self.offsets)%n
        values=message[...,None]*edge*mass[:,None,None]/math.sqrt(e)
        address=((b_idx[:,None,None]*c.regions+regions[:,None,None])*n+dest).reshape(-1)
        delta=torch.zeros_like(state.reshape(-1)).scatter_add(0,address,values.reshape(-1)).view_as(state)
        state=c.state_decay*state+delta
        accum=accumulator_step(accum,delta,c.accumulation if accumulation is None else accumulation,c.accumulator_decay)
        candidates=state[b_idx[:,None,None],regions[:,None,None],dest]
        selected=candidates.abs().argmax(-1,keepdim=True)
        next_idx=dest.gather(-1,selected).squeeze(-1)
        next_message=torch.tanh(state[b_idx[:,None],regions[:,None],next_idx])
        acc_message=accum[b_idx[:,None],regions[:,None],next_idx]
        return state,accum,next_idx,next_message,acc_message,int(values.numel()),int((edge.detach()!=0).sum())

    def _router_logits(self,regions,previous,message,fatigue,acc_message,cycle,fatigue_strength=None):
        c=self.c
        previous_normalized=previous.clamp_min(0).float()/c.regions
        features=torch.cat((self.region_emb[regions],message.mean(-1,keepdim=True),
            message.abs().mean(-1,keepdim=True),fatigue[:,None],
            previous_normalized[:,None],acc_message.abs().mean(-1,keepdim=True),
            torch.full_like(fatigue[:,None],cycle/max(c.max_cycles,1))),dim=-1)
        base=self.router(features)+self.transition_logits[regions]
        base=base.clone()
        base[previous<0,c.gateways+1]=-30
        if not c.recurrence:
            base[:,0]=-30
            base[:,c.gateways+1]=-30
            # A strictly increasing gateway destination makes the soft
            # region-mass surrogate acyclic when recurrence is disabled.
            invalid=self.gateway_table[regions]<=regions[:,None]
            base[:,1:c.gateways+1]=base[:,1:c.gateways+1].masked_fill(invalid,-30)
        effective=base.clone()
        if c.fatigue:
            strength=c.fatigue_strength if fatigue_strength is None else fatigue_strength
            effective[:,0]-=strength*fatigue
            effective[:,c.gateways+1]-=strength*fatigue
        return base.softmax(-1),effective.softmax(-1),effective

    def _hard_token(self,token_ids,state,fatigue,mode,generator,trace,forced_region=None,
                    recurrence=None,accumulation=None,fatigue_on=None,max_cycles=None,
                    recurrence_override=None,fatigue_strength=None,recovery=None):
        c=self.c;b=token_ids.shape[0];w=c.initial_regions;r=c.regions;n=c.region_size
        token=self.emb(token_ids)
        global_logits=self.global_router(token)
        if forced_region is not None:
            global_logits=global_logits.clone();global_logits[:,int(forced_region)]+=20
        current=global_logits.topk(w,dim=-1).indices
        initial_strength=torch.sigmoid(global_logits.gather(1,current))
        indices=self._initial_indices(token_ids[:,None],current)
        message=self.seed_proj(token)[:,None,:]*initial_strength[...,None]
        previous=torch.full_like(current,-1)
        alive=torch.ones_like(current,dtype=torch.bool)
        visited=torch.zeros(b,w,r,device=token_ids.device,dtype=torch.bool)
        accum=torch.zeros_like(state)
        readout=torch.zeros(b,c.d_model,device=token_ids.device)
        steps=[];edge_visits=0;nonzero_edge_visits=0;region_visits=torch.zeros(b,r,device=token_ids.device)
        visited_regions=[];base_local=[];expected_visits=[];acc_magnitudes=[];contributions=[]
        cycles=max_cycles or c.max_cycles
        for cycle in range(cycles):
            bi,wi=alive.nonzero(as_tuple=True)
            if bi.numel()==0: break
            region=current[bi,wi]; old_prev=previous[bi,wi]
            visited=visited.index_put((bi,wi,region),torch.ones_like(region,dtype=torch.bool))
            idx=indices[bi,wi];msg=message[bi,wi]
            region_visits=region_visits.flatten().scatter_add(0,bi*r+region,
                torch.ones_like(region,dtype=region_visits.dtype)).view(b,r)
            old_acc=accum[bi[:,None],region[:,None],idx].abs().mean(-1)
            state,accum,next_idx,next_msg,acc_msg,used,nonzero_used=self._propagate(
                state,accum,bi,region,idx,msg,torch.ones_like(region,dtype=msg.dtype),accumulation)
            edge_visits+=used;nonzero_edge_visits+=nonzero_used
            tired=fatigue[bi,region]
            base_prob,prob,route_logits=self._router_logits(region,old_prev,next_msg,tired,acc_msg,cycle,fatigue_strength)
            if recurrence_override is not None:
                p=max(1e-4,min(1-1e-4,float(recurrence_override)))
                route_logits=route_logits.clone()
                route_logits[:,0]=math.log(p/(1-p))+torch.logsumexp(route_logits[:,1:],dim=-1)
                prob=route_logits.softmax(-1)
            if recurrence is False or (recurrence is None and not c.recurrence):
                route_logits=route_logits.clone();route_logits[:,0]=-30
                route_logits[:,c.gateways+1]=-30
                for gateway in range(c.gateways):
                    target=self.gateway_table[region,gateway]
                    route_logits[visited[bi,wi,target],gateway+1]=-30
                prob=route_logits.softmax(-1)
            if fatigue_on is False and c.fatigue:
                route_logits=route_logits.clone()
                strength=c.fatigue_strength if fatigue_strength is None else fatigue_strength
                route_logits[:,0]+=strength*tired
                route_logits[:,c.gateways+1]+=strength*tired
                prob=route_logits.softmax(-1)
            if mode=='train':
                route=F.gumbel_softmax(route_logits,tau=1.,hard=True)
                action=route.argmax(-1)
            elif mode=='sampled':
                action=torch.multinomial(prob,1,generator=generator).squeeze(-1)
                route=F.one_hot(action,c.gateways+3).float()
            else:
                action=prob.argmax(-1)
                route=F.one_hot(action,c.gateways+3).float()
            gateway_action=(action>=1)&(action<=c.gateways)
            returning=action==c.gateways+1
            outputting=action==c.gateways+2
            gateway_index=(action-1).clamp(0,c.gateways-1)
            gateway_dest=self.gateway_table[region,gateway_index]
            destination=torch.where(gateway_action,gateway_dest,torch.where(returning,old_prev,region))
            destination=torch.where(outputting,torch.full_like(destination,-1),destination)
            contribution=self.readout_proj(acc_msg if (accumulation is None and c.accumulation) or accumulation else next_msg)
            output_gate=route[:,c.gateways+2]
            readout=readout.index_add(0,bi,contribution*output_gate[:,None])
            route_gain=route.gather(1,action[:,None]).squeeze(-1)
            indices=indices.index_put((bi,wi),next_idx)
            message=message.index_put((bi,wi),next_msg*route_gain[:,None])
            current=current.index_put((bi,wi),destination.clamp_min(0))
            new_prev=torch.where(action==0,old_prev,region)
            previous=previous.index_put((bi,wi),new_prev)
            alive=alive.index_put((bi,wi),~outputting)
            decay=c.recovery if recovery is None else recovery
            fatigue=fatigue*decay if (fatigue_on is None and c.fatigue) or fatigue_on else fatigue*0
            usage=torch.zeros_like(fatigue).flatten().scatter_add(0,bi*r+region,
                torch.ones_like(region,dtype=fatigue.dtype)).view_as(fatigue)
            fatigue=fatigue+usage if (fatigue_on is None and c.fatigue) or fatigue_on else fatigue
            visited_regions.extend(region.detach().cpu().tolist())
            base_local.extend(base_prob[:,0].detach().cpu().tolist())
            expected_visits.extend((1-prob[:,-1]).detach().cpu().tolist())
            acc_magnitudes.extend(acc_msg.abs().mean(-1).detach().cpu().tolist())
            contributions.extend((contribution.norm(dim=-1)*output_gate).detach().cpu().tolist())
            if trace:
                events=[]
                for j in range(region.numel()):
                    if int(bi[j])!=0: continue
                    events.append({'region':int(region[j]),'previous_region':int(old_prev[j]),
                        'destination':int(destination[j]),'action':int(action[j]),
                        'base_probability':base_prob[j].detach().cpu().tolist(),
                        'route_probability':prob[j].detach().cpu().tolist(),
                        'accumulator_before':float(old_acc[j]),
                        'accumulator_magnitude':float(acc_msg[j].abs().mean()),
                        'contribution_magnitude':float(contribution[j].norm()*output_gate[j]),
                        'walker':int(wi[j]),
                        'fatigue':float(tired[j]),'active_nodes':c.active_nodes,
                        'node_indices':idx[j].detach().cpu().tolist(),
                        'nonzero_edges':int((self.local_edge[region[j],idx[j]].detach().abs()>.33).sum())})
                steps.append({'cycle':cycle,'events':events,'active':[e['region'] for e in events],
                    'fatigue':fatigue[0].detach().cpu().tolist(),
                    'accumulator_magnitude':accum[0].abs().mean(-1).detach().cpu().tolist(),
                    'region_visits':region_visits[0].detach().cpu().tolist()})
        if alive.any():
            bi,wi=alive.nonzero(as_tuple=True)
            region=current[bi,wi];idx=indices[bi,wi]
            last=accum[bi[:,None],region[:,None],idx] if (accumulation is None and c.accumulation) or accumulation else message[bi,wi]
            readout=readout.index_add(0,bi,self.readout_proj(last))
        logits=self.head(self.norm(readout/w))/math.sqrt(c.d_model)
        diagnostics={'edge_visits':edge_visits,'nonzero_edge_visits':nonzero_edge_visits,
            'region_visits':region_visits.detach().cpu().tolist(),
            'visited_regions':visited_regions,'base_local_probability':base_local,
            'expected_nonoutput_probability':expected_visits,'accumulator_magnitudes':acc_magnitudes,
            'contribution_magnitudes':contributions}
        return logits,state,fatigue,steps,diagnostics

    def _expected_token(self,token_ids,state,fatigue,trace=False):
        """Soft region-mass surrogate, without sampled categorical decisions."""
        c=self.c;b=token_ids.shape[0];r=c.regions;k=c.active_nodes
        token=self.emb(token_ids);gl=self.global_router(token)
        selected=gl.topk(c.initial_regions,dim=-1).indices
        mass=torch.zeros_like(gl).scatter(1,selected,torch.sigmoid(gl.gather(1,selected)))
        regions=torch.arange(r,device=token_ids.device)[None].expand(b,-1)
        indices=self._initial_indices(token_ids[:,None],regions)
        message=self.seed_proj(token)[:,None,:].expand(-1,r,-1)
        origin=torch.eye(r,device=token_ids.device)[None].expand(b,-1,-1)
        valid_prev=torch.zeros_like(mass)
        accum=torch.zeros_like(state);readout=torch.zeros(b,c.d_model,device=token_ids.device)
        for cycle in range(c.max_cycles):
            bi,ri=(mass>1e-7).nonzero(as_tuple=True)
            if bi.numel()==0:break
            state,accum,idx,msg,acc_msg,_,_=self._propagate(state,accum,bi,ri,indices[bi,ri],message[bi,ri],mass[bi,ri])
            next_indices=indices.index_put((bi,ri),idx)
            next_message=message.index_put((bi,ri),msg)
            prev_id=(origin[bi,ri]*torch.arange(r,device=token_ids.device)).sum(-1)
            prev_id=torch.where(valid_prev[bi,ri]>1e-6,prev_id,torch.full_like(prev_id,-1)).long()
            base,prob,_=self._router_logits(ri,prev_id,msg,fatigue[bi,ri],acc_msg,cycle)
            probs=torch.zeros(b,r,c.gateways+3,device=token_ids.device).index_put((bi,ri),prob)
            readout=readout.index_add(0,bi,self.readout_proj(acc_msg)*mass[bi,ri,None]*prob[:,-1,None])
            next_mass=mass*probs[:,:,0]
            next_msg_mass=next_message*next_mass[:,:,None]
            next_orig_mass=origin*next_mass[:,:,None]
            next_valid_mass=valid_prev*next_mass
            for gateway in range(c.gateways):
                destination=self.gateway_table[:,gateway][None].expand(b,-1)
                flux=mass*probs[:,:,gateway+1]
                next_mass=next_mass.scatter_add(1,destination,flux)
                next_msg_mass=next_msg_mass.scatter_add(1,destination[:,:,None].expand(-1,-1,k),next_message*flux[:,:,None])
                address=(torch.arange(b,device=mass.device)[:,None]*r+destination).reshape(-1)
                source_onehot=F.one_hot(regions,r).float()*flux[:,:,None]
                next_orig_mass=next_orig_mass.reshape(b*r,r).index_add(0,address,source_onehot.reshape(b*r,r)).view(b,r,r)
                next_valid_mass=next_valid_mass.scatter_add(1,destination,flux)
            return_flux=origin*(mass*probs[:,:,c.gateways+1])[:,:,None]
            next_mass=next_mass+return_flux.sum(1)
            next_msg_mass=next_msg_mass+torch.einsum('bsd,bsk->bdk',return_flux,next_message)
            next_orig_mass=next_orig_mass+return_flux.transpose(1,2)
            next_valid_mass=next_valid_mass+return_flux.sum(1)
            message=next_msg_mass/next_mass.clamp_min(1e-8)[:,:,None]
            origin=next_orig_mass/next_mass.clamp_min(1e-8)[:,:,None]
            valid_prev=next_valid_mass/next_mass.clamp_min(1e-8)
            mass=next_mass
            indices=next_indices
            fatigue=c.recovery*fatigue+mass if c.fatigue else torch.zeros_like(fatigue)
        if mass.any():
            # Remaining route mass terminates at the configured cycle bound.
            node_values=accum.gather(2,indices)
            readout=readout+(self.readout_proj(node_values)*mass[:,:,None]).sum(1)
        return self.head(self.norm(readout/c.initial_regions))/math.sqrt(c.d_model),state,fatigue

    def forward(self,x,trace=False,routing_mode=None,stochastic=None,seed=None,
                recurrence=None,accumulation=None,fatigue=None,max_cycles=None,forced_region=None,
                recurrence_override=None,fatigue_strength=None,recovery=None,**_):
        c=self.c;b,t=x.shape
        if t>c.context:raise ValueError('context limit exceeded')
        mode=routing_mode or ('train' if self.training else 'sampled' if stochastic is not False else 'argmax')
        generator=None
        if seed is not None:
            generator=torch.Generator(device=x.device);generator.manual_seed(seed)
        state=torch.zeros(b,c.regions,c.region_size,device=x.device)
        tired=torch.zeros(b,c.regions,device=x.device)
        outputs=[];timeline=[];diagnostics=[]
        for ti in range(t):
            if mode=='expected':
                logits,state,tired=self._expected_token(x[:,ti],state,tired)
                steps=[];diag={}
            else:
                logits,state,tired,steps,diag=self._hard_token(x[:,ti],state,tired,mode,generator,trace,
                    forced_region,recurrence,accumulation,fatigue,max_cycles,
                    recurrence_override,fatigue_strength,recovery)
            outputs.append(logits)
            if trace:timeline.append(steps)
            diagnostics.append(diag)
        self.last_diagnostics=diagnostics
        return torch.stack(outputs,1),timeline if trace else None
