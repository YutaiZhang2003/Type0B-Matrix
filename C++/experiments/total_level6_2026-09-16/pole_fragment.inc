    const Pole &pole(int edge, int shift, int r, int s) {
        std::array<int, 4> key{edge, shift, r, s};
        auto old = geometry_.find(key);
        if (old != geometry_.end())
            return old->second;
        S h = weight(edge, shift),
          rad = S((r - s) * (r - s)) + S(4 * (r * s - 1)) * h + S(4) * h * h,
          base = S(r * s - 1) + S(2) * h;
        S x1 = (base + root(rad)) / S(1 - r * r), x2 = (base - root(rad)) / S(1 - r * r),
          x = magnitude(x1) >= magnitude(x2) ? x1 : x2;
        require(x != S(0), "CCY pole at infinity");
        S c = S(13) + S(6) * (x + S(1) / x), num = -S(12) * power(x, 2 * r * s - 1),
          den = S(1 - r * r) * x * x - S(1 - s * s);
        std::vector<std::pair<int, int>> factors;
        for (int a = 1 - r; a <= r; a++)
            for (int b = 1 - s; b <= s; b++)
                if ((a || b) && !(a == r && b == s))
                    factors.emplace_back(a, b);
        for (int target : {-1, 1}) {
            auto it = std::find_if(factors.begin(), factors.end(),
                                   [&](auto p) { return p.first && p.second == p.first * target; });
            if (it == factors.end())
                num *= x + S(target);
            else {
                den *= S(it->first);
                factors.erase(it);
            }
        }
        for (auto [a, b] : factors)
            den *= S(a) * x + S(b);
        require(den != S(0), "confluent Kac residue requires a generic-weight limit");
        int id = 0;
        while (id < int(centers_.size()) && centers_[id] != c)
            id++;
        if (id == int(centers_.size()))
            centers_.push_back(c);
        return geometry_.emplace(key, Pole{c, num / den, x, id}).first->second;
    }
    S fusion(const Pole &p, int r, int s, int top, int ts, int bottom, int bs) {
        std::array<int, 7> key{p.id, r, s, top, ts, bottom, bs};
        auto old = fusion_.find(key);
        if (old != fusion_.end())
            return old->second;
        S a = weight(top, ts), b = weight(bottom, bs), x = p.x,
          lam = x + S(2) + S(1) / x - S(4) * a, diff = S(4) * (b - a), v(1);
        for (int u = 1 - r; u < r; u += 2)
            for (int t = 1 - s; t < s; t += 2) {
                if (!u && !t)
                    v *= b - a;
                else if (u > 0 || (!u && t > 0)) {
                    S shift = S(u * u) * x + S(2 * u * t) + S(t * t) / x;
                    v *= (power(diff + shift, 2) - S(4) * lam * shift) / S(16);
                }
            }
        fusion_[key] = v;
        return v;
    }
