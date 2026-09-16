#pragma once
// Mercedes through total level six: primitive lengths three and four suffice.
// Expand each lifted primitive contribution, not a fitted/pbw vacuum series.
// The first fermionic oscillator has h=3/2; the first bosonic oscillator h=2.
// Higher modes and products of two nonconstant primitive factors exceed level six.
inline QPoly ns_mercedes_schottky(const Graph&g,int level){
    require(g.edges==6&&g.slots.size()==4&&level<=6,"Mercedes NS Schottky implementation supports total level <=6");
    auto one=constant(1);if(level<5)return one;
    std::array<QMat,3>coord{QMat{QPoly{},one,one,{}},QMat{one,constant(-1),{},one},QMat{one,{},{},one}};
    auto back=coord;back[1]={one,one,{},one};std::vector<QMat>maps;
    for(int e=0;e<g.edges;e++)for(int d=0;d<2;d++){Key k{};k[e]=1;QMat inv{QPoly{},QPoly{{k,Rational(1)}},one,{}};maps.push_back(multiply(back[g.ends[e][1-d]%3],multiply(inv,coord[g.ends[e][d]%3],level),level));}
    std::set<Part>classes;Part word;int maximum=2*level/3;
    std::function<void(int,int)>visit=[&](int start,int v){int n=word.size();if(n&&v==start&&word.back()!=(word.front()^1)){
        bool primitive=true;for(int p=1;p<n;p++)if(n%p==0){bool same=true;for(int j=0;j<n;j++)if(word[j]!=word[j%p])same=false;if(same)primitive=false;}
        if(primitive){Part inv;for(auto it=word.rbegin();it!=word.rend();++it)inv.push_back(*it^1);Part best=word;for(auto w:{word,inv})for(int r=0;r<n;r++){Part rot(w.begin()+r,w.end());rot.insert(rot.end(),w.begin(),w.begin()+r);best=std::min(best,rot);}classes.insert(best);}
    }if(n==maximum)return;for(int e=0;e<g.edges;e++)for(int d=0;d<2;d++)if(g.ends[e][d]/3==v){int arc=2*e+d;if(n&&arc==(word.back()^1))continue;word.push_back(arc);visit(start,g.ends[e][1-d]/3);word.pop_back();}};
    for(int v=0;v<int(g.slots.size());v++)visit(v,v);
    QPoly out=one;std::ofstream audit("schottky_seed_L"+std::to_string(level)+".jsonl");
    for(auto w:classes){Key base{};int cycle=0;QMat m{one,{},{},one};for(int arc:w){base[arc/2]+=3;cycle^=1<<(arc/2);m=multiply(maps[arc],m,level);}auto trace=add(m[0],m[3]);auto inv=inverse(trace,level);auto normalized=scale(inv,trace.at(Key{}));
        // k=det(M)/tr(M)^2+O(det(M)^2).  Therefore the h=3/2
        // contribution has the normalized trace factor (tr(M)(0)/tr(M))^3.
        // The lift in this ordered PBW frame is fixed by the three-point
        // supercurrent two-point forms and the graph permutation.
        auto cube=multiply(multiply(normalized,normalized,level),normalized,level);
        for(auto[k,v]:cube){Key n=base;for(int e=0;e<6;e++)n[e]+=2*k[e];if(total(n)<=2*level)out[n]+=Rational(g.sign_of(cycle))*v;}
        if(2*int(w.size())<=level){Key n{};for(int arc:w)n[arc/2]+=4;out[n]+=1;}
        audit<<"{\"arcs\":[";for(size_t i=0;i<w.size();i++){if(i)audit<<',';audit<<w[i];}audit<<"],\"cycle_mask\":"<<cycle<<",\"fermion_sign\":"<<g.sign_of(cycle)<<"}\n";
    }
    return out;
}
