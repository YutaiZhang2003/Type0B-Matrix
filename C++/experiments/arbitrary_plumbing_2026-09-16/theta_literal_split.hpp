#pragma once
#include "direct_pbw.hpp"
namespace ramond {
template<class S> class ThetaSplitPBW {
    ScaWords<S> words_;
    std::array<std::unique_ptr<ScaModule<S>>,3> gram_modules_,ward_modules_;
    std::unique_ptr<ScaWard<S>> forms_[2];
    struct Edge { std::vector<int> states; std::vector<S> inverse; };
    std::map<std::array<int,3>,Edge> edges_;
    std::array<S,3> phases_;
    int p_,f_;
    bool opposite_;
    S insertion_scalar_;
    FermionForm fermion_form_;
    const Edge &edge(int slot,int level2,int parity) {
        std::array<int,3> key{slot,level2,parity};
        auto found=edges_.find(key); if(found!=edges_.end()) return found->second;
        double start=seconds(); Edge result;
        int level=slot ? level2/2 : level2;
        for(int l=0;l<=(slot?level:level/2);l++)
            for(const auto &ls:partitions(l))
                for(const auto &gs:partitions(level-(slot?l:2*l),true,!slot)) {
                    ScaWord word;
                    for(int n:ls) word.push_back({0,-2*n});
                    for(int n:gs) word.push_back({1,slot ? -2*n : -n});
                    int id=words_.intern(word),g=slot ? (parity+gs.size())%2 : 0;
                    if((words_.words[id].parity+g)%2==parity) result.states.push_back(2*id+g);
                }
        times.metadata+=seconds()-start; start=seconds();
        int n=static_cast<int>(result.states.size()); std::vector<S> gram(size_t(n)*n);
        for(int i=0;i<n;i++) for(int j=0;j<=i;j++) {
            S v=gram_modules_[slot]->inner(result.states[i],result.states[j]);
            gram[i*n+j]=v; gram[j*n+i]=v; counts.gram_entries++;
        }
        times.gram_entries+=seconds()-start; start=seconds();
        result.inverse=pbw_inverse(std::move(gram),n);
        times.gram_inverse+=seconds()-start; counts.gram_matrices++;
        return edges_.emplace(key,std::move(result)).first->second;
    }
    S contract(const std::vector<S> &left,const std::array<const Edge*,3> &edges,const std::vector<S> &right) {
        std::array<size_t,3> d{edges[0]->states.size(),edges[1]->states.size(),edges[2]->states.size()};
        size_t volume=d[0]*d[1]*d[2];
        std::vector<S> current=right,next(volume); S scratch;
        // Each axis contraction writes directly in (a,b,c) order. No cached permutations.
        for(int axis=0;axis<3;axis++) {
            size_t stride=axis==0 ? d[1]*d[2] : axis==1 ? d[2] : 1;
            size_t n=d[axis]; const auto &inverse=edges[axis]->inverse;
            for(size_t base=0;base<volume;base+=n*stride)
                for(size_t offset=0;offset<stride;offset++)
                    for(size_t i=0;i<n;i++) {
                        S &v=next[base+i*stride+offset]; v=S(0);
                        for(size_t j=0;j<n;j++) pbw_madd(v,inverse[i*n+j],current[base+j*stride+offset],scratch);
                    }
            current.swap(next);
        }
        S answer(0);
        for(size_t i=0;i<volume;i++) pbw_madd(answer,left[i],current[i],scratch);
        counts.contraction_products+=volume*(d[0]+d[1]+d[2]+1);
        counts.contractions++; return answer;
    }
  public:
    struct Timings { double metadata=0,gram_entries=0,gram_inverse=0,vertices=0,contractions=0; } times;
    struct Counts { size_t gram_entries=0,gram_matrices=0,vertex_entries=0,reused_vertex_tensors=0,
                          contractions=0,contraction_products=0; } counts;
    ThetaSplitPBW(S b,const std::array<S,3> &p,int primary_parity,int form_parity,int eta,bool opposite)
        :p_(primary_parity),f_(form_parity),opposite_(opposite) {
        S q=b+S(1)/b,c=rational<S>(3,2)+S(3)*q*q;
        insertion_scalar_=q/root(S(2));
        for(int j=0;j<3;j++) {
            S h=q*q/S(8)-p[j]*p[j]/S(2)+(j?rational<S>(1,16):S(0));
            gram_modules_[j]=std::make_unique<ScaModule<S>>(words_,h,c,p[j]/root(S(2)),j!=0,true);
            ward_modules_[j]=std::make_unique<ScaModule<S>>(words_,h,c,p[j]/root(S(2)),j!=0,false);
        }
        std::array<ScaModule<S>*,3> modules{ward_modules_[0].get(),ward_modules_[1].get(),ward_modules_[2].get()};
        forms_[0]=std::make_unique<ScaWard<S>>(words_,modules,p_,f_,eta);
        if(opposite_) forms_[1]=std::make_unique<ScaWard<S>>(words_,modules,p_,f_,-eta);
        for(int k=0;k<3;k++) phases_[k]=power((S(-1)+S(Machine(0,1)))/root(S(2)),k);
    }

    // Explicit four-edge tensor sewing: the physical middle state is unchanged,
    // whereas the auxiliary states are connected by Q Theta psi_{lL-lR}.
    // No physical-times-fermion convolution is used to build this numerator.
    std::array<S,8> coefficient(Index total,bool insert=true) {
        require(opposite_, "split probe requires opposite vertex signs");
        std::array<S,8> result{};
        for(int pa=0;pa<=total[0];pa++)
        for(int pb=0;pb<=2*std::min(total[1],total[2]);pb+=2)
        for(int pc=0;pc<=total[3];pc+=2) {
            const int aa=total[0]-pa,bl=total[1]-pb/2,br=total[2]-pb/2,ac=(total[3]-pc)/2;
            for(int parity=0;parity<2;parity++) {
                std::array<int,3> pr{pa%2,parity,(f_+pa+parity)%2};
                std::array<const Edge*,3> edges{&edge(0,pa,pr[0]),&edge(1,pb,pr[1]),&edge(2,pc,pr[2])};
                size_t volume=edges[0]->states.size()*edges[1]->states.size()*edges[2]->states.size();
                if(!volume)continue;
                std::vector<S> physical[2];
                for(int v=0;v<2;v++)
                    for(int a:edges[0]->states)for(int b:edges[1]->states)for(int c:edges[2]->states)
                        physical[v].push_back(phases_[b%2+c%2]*forms_[v]->value({a,b,c}));
                for(const auto &A:partitions(aa,true,true))
                for(const auto &BL:partitions(bl,true,false))
                for(const auto &BR:partitions(br,true,false))
                for(const auto &C:partitions(ac,true,false))
                for(int gl=0;gl<2;gl++) {
                    int gr=(BL.size()+BR.size()+gl)%2;
                    int cg=(A.size()+BL.size()+C.size()+gl)%2;
                    int mode=bl-br;
                    S matrix(0);
                    if(!insert) {
                        if(mode!=0 || BL!=BR || gl!=gr)continue;
                        matrix=S(1);
                    } else if(mode==0) {
                        if(BL!=BR || gl!=gr)continue;
                        matrix=insertion_scalar_*S(sign(gl));
                    } else {
                        auto action=aux_act(1,mode,AuxState{BL,gl});
                        if(action.second==0 || action.first.modes!=BR || 1-action.first.ground!=gr)continue;
                        matrix=insertion_scalar_*root(S(2))*from_rational<S>(action.second)
                            *S(sign(int(BR.size())+gl+1));
                    }
                    S left=fermion_form_.complex_value<S>({AuxState{A,0},AuxState{BL,gl},AuxState{C,cg}});
                    S right=fermion_form_.complex_value<S>({AuxState{A,0},AuxState{BR,gr},AuxState{C,cg}});
                    if(left==S(0)||right==S(0))continue;
                    int ap=A.size()%2,bp=(BL.size()+gl)%2,cp=(C.size()+cg)%2;
                    int koszul=((pr[0]+p_)%2)*(bp+cp)+pr[1]*cp;
                    std::vector<S> vertices[2];
                    for(const auto &x:physical[0])vertices[0].push_back(S(sign(koszul))*left*x);
                    for(const auto &x:physical[1])vertices[1].push_back(S(sign(koszul))*right*x);
                    int index=((pr[0]+p_+ap)%2)|((pr[1]^bp)<<1)|((pr[2]^cp)<<2);
                    result[index]+=S(sign(ap)*theta_sign(index))*matrix*contract(vertices[0],edges,vertices[1]);
                }
            }
        }
        return result;
    }
    size_t ward_entries() const { return forms_[0]->cache_size()+(opposite_?forms_[1]->cache_size():0); }
    size_t ward_hits() const { return forms_[0]->hits+(opposite_?forms_[1]->hits:0); }
    size_t ward_terms() const { return forms_[0]->action_terms+(opposite_?forms_[1]->action_terms:0); }
    size_t ward_buckets() const { return forms_[0]->cache_buckets()+(opposite_?forms_[1]->cache_buckets():0); }
    size_t word_count() const { return words_.words.size(); }
    size_t action_entries() const {
        size_t n=0; for(int j=0;j<3;j++) n+=gram_modules_[j]->cache_size()+ward_modules_[j]->cache_size(); return n;
    }
};
} // namespace ramond
