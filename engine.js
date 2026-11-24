/* =============================================================
   我知道你几岁（中国记忆版）核心贝叶斯推理引擎
   -------------------------------------------------------------
   ・针对 6〜85 岁中国大陆用户，结合 6 个隐式画像维度（共 64 种组合）：
       0 retro（怀旧/爱考古）
       1 parent（陪娃看过儿童内容）
       2 game（爱打游戏）
       3 girl（偏女生向文化）
       4 out（户外/体育偏好）
       5 tv（童年家里常看电视）
   ・动态两阶段出题（类 Akinator 信息增益最大化）：
       前半（粗筛）：80% 置信区间 > coarse 时，按 coarseBin 岁分桶最大化互信息
       后半（精筛）：进入 1 岁粒度，对年代分界清晰的「生活经历题（pick）」加权
       若后半遇到极低概率的意外回答（pred < surprise），自动放宽分布并短暂退回粗筛
   ・对某一领域频繁选「不知道」的用户自动降低该领域权重并切换题型，
     适时追问童年娱乐习惯（prof）或家庭视听限制（home），不靠单一兴趣硬猜
   ============================================================= */
(function (root, factory) {
  const api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  else root.NenreiEngine = api;
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const AMIN = 6;
  const AMAX = 85;
  const N = AMAX - AMIN + 1;
  const AGES = Array.from({ length: N }, (_, i) => AMIN + i);
  const BIN = 5;
  const NBIN = Math.ceil(N / BIN);
  const binOf = (age) => Math.min(NBIN - 1, Math.max(0, Math.floor((age - AMIN) / BIN)));

  const sig = (x) => 1 / (1 + Math.exp(-x));
  const win = (a, a0, a1, s) => sig((a - (a0 - 0.5)) / s) * sig(((a1 + 0.5) - a) / s);

  function ncdf(x) {
    const t = 1 / (1 + 0.2316419 * Math.abs(x));
    const d = 0.3989423 * Math.exp(-x * x / 2);
    const p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
    return x > 0 ? 1 - p : p;
  }

  const norm = (arr) => {
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += arr[i];
    if (!(s > 0)) return arr.map(() => 1 / arr.length);
    return arr.map((v) => v / s);
  };

  function mixNoise(p, amount) {
    const u = 1 / p.length;
    return norm(p.map((v) => v * (1 - amount) + amount * u));
  }

  /* ---------------------------------------------------------
     流行记忆题（know）：[超迷/常用, 玩过/看过, 只听过名字, 不知道]
     --------------------------------------------------------- */
  function knowProbs(p, age, now, ty) {
    const B = now - age - 0.5;
    const y0 = p.y;
    const y1 = p.y + (p.sp || 1);
    const a0 = p.a[0];
    const a1 = p.a[1];
    const steps = Math.max(2, Math.ceil((y1 - y0) / 0.25));
    const dt = (y1 - y0) / steps;
    let acc = 0;
    for (let k = 0; k <= steps; k++) {
      const t = y0 + k * dt;
      acc += win(t - B, a0, a1, 0.7) * (k === 0 || k === steps ? dt / 2 : dt);
    }
    const D = 1 - Math.exp(-acc / 0.9);
    const ageEnd = y1 - B;
    const ageMid = (y0 + y1) / 2 - B;
    const O = sig((ageMid - (a1 + 0.5)) / 1.0);
    const Y = sig(((a0 - 0.5) - ageEnd) / 1.0);
    const oy = O + Y;
    const wIn = D;
    const wOld = (1 - D) * (oy > 1e-9 ? O / oy : 0.5);
    const wYoung = (1 - D) * (oy > 1e-9 ? Y / oy : 0.5);

    const fh = ty && ty.fanH != null ? ty.fanH : 1;
    const fa = ty && ty.fanA != null ? ty.fanA : 1;
    const h = Math.min(0.95, p.h * fh);
    const aw0 = p.aw != null ? p.aw : (fa !== 1 ? 0.94 : null);
    const aw = aw0 != null ? Math.max(h + 0.02, Math.min(0.99, aw0 * fa)) : null;
    const pin = aw == null
      ? [h, (1 - h) * 0.72, (1 - h) * 0.22, (1 - h) * 0.06]
      : [h, (aw - h) * 0.72, (aw - h) * 0.28, 1 - aw];

    const gap = Math.max(0, ageMid - a1);
    const fn = 1 + (fa - 1) * 0.6;
    const ho = Math.min(0.9, (p.ho != null ? p.ho : 0.04) * fh);
    const ko = Math.max(ho, Math.min(0.95, (p.ko != null ? p.ko : 0.35) * fa) * Math.exp(-gap / 45));
    const no = Math.max(ko, Math.min(0.98, (p.no != null ? p.no : 0.85) * fn * Math.exp(-gap / 70)));
    const pold = [ho, ko - ho, no - ko, 1 - no];

    // 陪娃效应：儿童向内容（a1 <= 13）在其流行时正值育儿期（25〜46岁）的人群也较了解
    if (a1 <= 13 && (!ty || ty.parent)) {
      const wp = win(ageMid, 25, 46, 2.5) * (p.pp != null ? p.pp : 1) * (ty ? 1 : 0.45);
      if (wp > 1e-3) {
        const hp = Math.max(ho, Math.min(0.9, (p.hp != null ? p.hp : 0.12) * fh));
        const kp = Math.max(ko, hp, Math.min(0.85, (p.ko != null ? p.ko : 0.35) * 2.2 * fa));
        const np = Math.max(no, kp, Math.min(0.97, ((p.no != null ? p.no : 0.85) + 0.1) * fn));
        const ppar = [hp, kp - hp, np - kp, 1 - np];
        for (let r = 0; r < 4; r++) pold[r] = pold[r] * (1 - wp) + ppar[r] * wp;
      }
    }

    const yrs = Math.max(0, (a0 - 0.5) - ageEnd);
    let R = Math.min(0.97, (p.r != null ? p.r : 0.3) * fa * Math.exp(-yrs / (p.rd || 12)));
    let hy = Math.min(0.3, (p.hy || 0.01) * fh * Math.exp(-yrs / 15));
    let rk = 0.35;
    if (ty && ty.retro) {
      R = Math.max(R, Math.min(0.85, 0.3 + 1.6 * (p.r != null ? p.r : 0.3)) * Math.exp(-yrs / ((p.rd || 12) * 2.5)));
      hy = Math.max(hy, 0.05 * Math.exp(-yrs / 30));
      rk = 0.5;
    }
    const pyoung = [hy, R * rk, R * (1 - rk), Math.max(0, 1 - hy - R)];
    const out = [0, 0, 0, 0];
    for (let r = 0; r < 4; r++) out[r] = wIn * pin[r] + wOld * pold[r] + wYoung * pyoung[r];
    return mixNoise(out, 0.08);
  }

  /** 4选1小测验（quiz）：[答对, 答错, 不知道] */
  function quizProbs(p, age, now, ty) {
    const k = knowProbs(p, age, now, ty);
    const K = Math.min(0.99, k[0] + k[1] * 0.92 + k[2] * 0.35);
    const g = p.g != null ? p.g : 0.25;
    const u = 0.5;
    const c = K * 0.95 + (1 - K) * (1 - u) * g;
    const w = K * 0.05 + (1 - K) * (1 - u) * (1 - g);
    const d = (1 - K) * u;
    return mixNoise([c, w, d], 0.04);
  }

  /** 学段划分 */
  const STAGES = {
    cn: [[-99, 6.5], [6.5, 12.5], [12.5, 15.5], [15.5, 18.5], [18.5, 999]],
  };

  function anchorProbs(p, age, now, stages, sd, ty) {
    const B = now - age - 0.5;
    const k = knowProbs(p, age, now, ty);
    const aware = 1 - k[3];
    const y0 = p.y;
    const y1 = p.y + (p.sp || 1);
    const steps = Math.max(2, Math.ceil((y1 - y0) / 0.25));
    const dt = (y1 - y0) / steps;
    const SD = sd || 1.3;
    const st = new Array(stages.length).fill(0);
    for (let k2 = 0; k2 <= steps; k2++) {
      const t = y0 + k2 * dt;
      const a = t - B;
      const wt = k2 === 0 || k2 === steps ? 0.5 : 1;
      for (let s = 0; s < stages.length; s++) {
        st[s] += wt * (ncdf((stages[s][1] - a) / SD) - ncdf((stages[s][0] - a) / SD));
      }
    }
    const sn = norm(st);
    const out = sn.map((v) => v * aware);
    out.push(1 - aware);
    return mixNoise(out, 0.04);
  }

  /** 时代生活经历题（pick） */
  const CHG_K = 0.7;
  function pickProbs(p, age, now) {
    const B = now - age - 0.5;
    const s = p.soft || 1.3;
    const K = 8;
    const rangesOf = (o) => [[o[1], o[2] == null ? now : o[2]]].concat((o[4] || []).map((x) => [x[0], x[1] == null ? now : x[1]]));
    let cs = null;
    if (p.stage) {
      const t0 = B + p.stage[0];
      const t1 = B + p.stage[1];
      cs = p.o.map((o, idx) => {
        const a = new Array(K).fill(0);
        if (idx === p.chg) return a;
        const ranges = rangesOf(o);
        for (let k = 0; k < K; k++) {
          const t = t0 + ((t1 - t0) * (k + 0.5)) / K;
          let c = 0;
          ranges.forEach(([lo, hi]) => { c += sig((t - lo) / (s * 0.6)) * sig((hi - t) / (s * 0.6)); });
          a[k] = Math.min(1, c);
        }
        return a;
      });
    }
    const m = p.o.map((o, idx) => {
      if (idx === p.chg) return 0;
      const w = o[3] != null ? o[3] : 1;
      if (cs) return (cs[idx].reduce((x, y) => x + y, 0) / K) * w + 0.01;
      let c = 0;
      rangesOf(o).forEach(([lo, hi]) => { c += win(B, lo, hi, s); });
      return Math.min(1, c) * w + 0.01;
    });
    let mn = norm(m);
    if (p.chg != null) {
      const pc = cs ? CHG_K * changeOf(p, cs, K, now) : 0.01;
      mn = mn.map((v, idx) => (idx === p.chg ? pc : v * (1 - pc)));
    }
    const dk = p.dk ? (p.dkp != null ? p.dkp : 0.07) : 0;
    const out = mn.map((v) => v * (1 - dk));
    if (p.dk) out.push(dk);
    return mixNoise(out, 0.03);
  }

  function changeOf(p, cs, K) {
    const st = [];
    p.o.forEach((o, idx) => {
      if (idx === p.chg || idx === p.none) return;
      const hi = o[2] == null ? 2050 : o[2];
      if (o[1] <= 1935 && hi >= 2045 && !(o[4] && o[4].length)) return;
      st.push(idx);
    });
    if (st.length < 2) return 0;
    const at = (ks) => {
      const q = st.map((idx) => {
        const w = p.o[idx][3] != null ? p.o[idx][3] : 1;
        return ks.reduce((x, k) => x + cs[idx][k], 0) * w;
      });
      const sq = q.reduce((x, y) => x + y, 0);
      return sq > 0.05 ? q.map((v) => v / sq) : null;
    };
    const a = at([0, 1]);
    const b = at([K - 2, K - 1]);
    if (!a || !b) return 0;
    let d = 0;
    for (let i = 0; i < st.length; i++) d += Math.abs(a[i] - b[i]);
    return Math.min(1, d / 2);
  }

  function pickProbsTy(p, age, now, ty) {
    const out = pickProbs(p, age, now);
    if (!ty || ty.fan == null) return out;
    const none = p.none != null ? p.none : (p.dk ? out.length - 1 : -1);
    if (none < 0) return out;
    const np = p.noneP != null ? p.noneP : 0.55;
    if (!ty.fan) return out.map((v, r) => v * (1 - np) + (r === none ? np : 0));
    const nf = p.noneFan != null ? p.noneFan : 0.4;
    const q = out.map((v, r) => (r === none ? v * nf : v));
    const sq = q.reduce((a, b) => a + b, 0);
    return q.map((v) => v / sq);
  }

  /** 童年玩乐偏好（prof）：中国家用游戏机（红白机/小霸王）约从 1978 年后出生者童年普及 */
  function profProbs(age, now, bits) {
    const B = now - age - 0.5;
    const A = sig((B - 1978) / 3.0);
    const G = (bits >> 2) & 1;
    const F = (bits >> 3) & 1;
    const O = (bits >> 4) & 1;
    const w = [A * (G ? 1.6 : 0.12) + 0.02, O ? 1.5 : 0.3, F ? 1.2 : 0.2, F ? 0.12 : 0.9, 0.35];
    return mixNoise(norm(w), 0.03);
  }

  /** 童年家庭视听管教（home）：中国家庭电视约从 1966 年后出生者童年普及，游戏约从 1978 年后 */
  function homeProbs(age, now, bits) {
    const B = now - age - 0.5;
    const TA = sig((B - 1966) / 3.5);
    const GA = sig((B - 1978) / 3.0);
    const G = (bits >> 2) & 1;
    const T = (bits >> 5) & 1;
    const both = T
      ? (G ? [0.5, 0.45, 0.03, 0.02] : [0.25, 0.37, 0.33, 0.05])
      : (G ? [0.06, 0.42, 0.04, 0.48] : [0.03, 0.18, 0.07, 0.72]);
    const tvOnly = T ? [0.55, 0.42, 0.0, 0.03] : [0.06, 0.34, 0.0, 0.6];
    const none = [0.12, 0.05, 0.0, 0.83];
    const w = [0, 1, 2, 3].map((r) => TA * (GA * both[r] + (1 - GA) * tvOnly[r]) + (1 - TA) * none[r]);
    const dk = 0.05;
    const out = norm(w).map((v) => v * (1 - dk));
    out.push(dk);
    return mixNoise(out, 0.03);
  }

  function arity(it) {
    if (it.t === 'know') return 4;
    if (it.t === 'quiz') return 3;
    if (it.t === 'anchor') return 6;
    if (it.t === 'pick') return it.o.length + (it.dk ? 1 : 0);
    return 1;
  }

  function likeOf(it, now, region, ty) {
    const rows = new Array(N);
    for (let i = 0; i < N; i++) {
      const age = AGES[i];
      if (it.t === 'know') rows[i] = knowProbs(it, age, now, ty);
      else if (it.t === 'quiz') rows[i] = quizProbs(it, age, now, ty);
      else if (it.t === 'anchor') rows[i] = anchorProbs(it, age, now, STAGES.cn, undefined, ty);
      else if (it.prof) rows[i] = profProbs(age, now, ty && ty.bits != null ? ty.bits : 0);
      else if (it.home) rows[i] = homeProbs(age, now, ty && ty.bits != null ? ty.bits : (1 << 2) | (1 << 5));
      else rows[i] = pickProbsTy(it, age, now, ty);
    }
    return rows;
  }

  /** 先验年龄分布（中文互联网玩家集中在 16〜38 岁，兼顾儿童与中老年尾部） */
  function basePrior() {
    const peak = 25;
    const p = AGES.map((a) => {
      const x = Math.log(Math.max(1, a - 4)) - Math.log(peak - 4);
      return Math.exp(-(x * x) / (2 * 0.58 * 0.58)) + 0.035;
    });
    return norm(p);
  }

  const H = (p) => {
    let h = 0;
    for (let i = 0; i < p.length; i++) if (p[i] > 1e-12) h -= p[i] * Math.log(p[i]);
    return h;
  };

  function model(opts) {
    const region = opts.region || 'cn';
    const now = opts.now;
    const R = Object.assign({
      minQ: 12, maxQ: 40, conf: 0.88, minGain: 0.015,
      coarse: 14, coarseBin: 6,
      lateFrom: 4, maxAnchor: 0, maxPick1: 2, maxPick: 8, pickBoost: 1.15,
      quizEvery: 4, quizMin: 0.3, temper: 0.8,
      retro: 0.15, parent: 0.45,
      game: 0.55, girl: 0.5, out: 0.5, tv: 0.94, traits: true,
      gateGame: 0.3, gateGirl: 0.3, gateOut: 0.3, gateTv: 0.3,
      profAfter: 3, homeAfter: 2, homeSurp: 0.25, genreMid: 0.75, genreLo: 0.4,
      surprise: 0.05, widen: 0.7, rewindQ: 2, maxRewind: 2,
    }, opts.rules || {});

    const items = opts.items.filter((it) => (!it.bank || it.bank === 'cn')
      && !(it.t === 'anchor' && !R.maxAnchor));

    const TP = [R.retro, R.parent, R.game, R.girl, R.out, R.tv];
    const NT = 1 << TP.length;
    const TW = [];
    for (let t = 0; t < NT; t++) {
      let w = 1;
      TP.forEach((p, b) => { w *= (t >> b) & 1 ? p : 1 - p; });
      TW.push(w);
    }

    const DOMS = {
      game: { bit: 2, fan: 1, p: R.game, lo: 0.2, alo: 0.8 },
      g: { bit: 3, fan: 1, p: R.girl, lo: 0.15, alo: 0.6 },
      b: { bit: 3, fan: 0, p: 1 - R.girl, lo: 0.15, alo: 0.6 },
      out: { bit: 4, fan: 1, p: R.out, lo: 0.35, alo: 0.85 },
      tv: { bit: 5, fan: 1, p: R.tv, lo: 0.15, alo: 0.55 },
      tvs: { bit: 5, fan: 1, p: R.tv, lo: 0.5, alo: 0.85 },
    };
    Object.values(DOMS).forEach((D) => {
      D.hi = (1 - (1 - D.p) * D.lo) / D.p;
      D.ahi = Math.min(1.6, (1 - (1 - D.p) * D.alo) / D.p);
    });

    const domsOf = (it) => (R.traits ? [].concat(it.dom || []).filter((d) => DOMS[d]) : []);

    function maskOf(it) {
      if (it.prof) return (1 << 2) | (1 << 3) | (1 << 4);
      if (it.home) return (1 << 2) | (1 << 5);
      let m = 0;
      if (it.t !== 'pick') {
        m |= 1;
        if (it.a && it.a[1] <= 13) m |= 2;
      }
      domsOf(it).forEach((d) => { m |= 1 << DOMS[d].bit; });
      return m;
    }

    function tyOf(it, t) {
      const ty = { retro: !!(t & 1), parent: !!(t & 2), fanH: 1, fanA: 1, fan: null, bits: t };
      domsOf(it).forEach((d) => {
        const D = DOMS[d];
        const fan = ((t >> D.bit) & 1) === D.fan;
        ty.fanH *= fan ? D.hi : D.lo;
        ty.fanA *= fan ? D.ahi : D.alo;
        ty.fan = ty.fan == null ? fan : ty.fan && fan;
      });
      return ty;
    }

    const L = new Map();
    items.forEach((it) => {
      const mask = maskOf(it);
      const byMv = new Map();
      for (let t = 0; t < NT; t++) {
        const mv = t & mask;
        if (!byMv.has(mv)) byMv.set(mv, likeOf(it, now, region, tyOf(it, mv)));
      }
      const byT = [];
      for (let t = 0; t < NT; t++) byT.push(byMv.get(t & mask));
      L.set(it.id, { mask, mvs: [...byMv.keys()], byMv, byT });
    });

    const profItem = items.find((it) => it.prof) || null;
    const homeItem = items.find((it) => it.home) || null;
    const prior = basePrior();
    const byId = new Map(items.map((it) => [it.id, it]));
    return { region, now, R, items, byId, NT, TW, domsOf, L, profItem, homeItem, prior };
  }

  function create(opts) {
    const M = opts.model || model(opts);
    const { R, items, NT, TW, domsOf, L, profItem, homeItem } = M;
    const rand = opts.rand || Math.random;
    const recent = new Set(opts.recent || []);
    const jitter = new Map();
    items.forEach((it) => { jitter.set(it.id, 0.72 + rand() * 0.56); });

    function powRows(LI) {
      if (!LI.byTp) {
        const pm = new Map();
        LI.byMv.forEach((rows, mv) => pm.set(mv, R.temper === 1 ? rows : rows.map((row) => row.map((v) => Math.pow(v, R.temper)))));
        const a = [];
        for (let t = 0; t < NT; t++) a.push(pm.get(t & LI.mask));
        LI.byTp = a;
      }
      return LI.byTp;
    }

    function traitNow() {
      const o = { game: 0, girl: 0, out: 0, tv: 0 };
      let tot = 0;
      for (let t = 0; t < NT; t++) {
        let s1 = 0;
        const pt = post[t];
        for (let i = 0; i < N; i++) s1 += pt[i];
        tot += s1;
        if ((t >> 2) & 1) o.game += s1;
        if ((t >> 3) & 1) o.girl += s1;
        if ((t >> 4) & 1) o.out += s1;
        if ((t >> 5) & 1) o.tv += s1;
      }
      if (tot > 0) Object.keys(o).forEach((k) => { o[k] /= tot; });
      return o;
    }

    let tr = null;
    function fits(it) {
      const ds = domsOf(it);
      if (!ds.length || !tr) return true;
      return ds.every((d) => (d === 'game' ? tr.game >= R.gateGame
        : d === 'g' ? tr.girl >= R.gateGirl
          : d === 'b' ? tr.girl <= 1 - R.gateGirl
            : d === 'out' ? tr.out >= R.gateOut
              : d === 'tv' ? tr.tv >= R.gateTv : true));
    }

    function tvGame(it) {
      return it.k === 'watch' || domsOf(it).some((d) => d === 'tv' || d === 'tvs' || d === 'game');
    }

    function genreOf(it) {
      const ds = domsOf(it);
      if (ds.includes('game')) return 'game';
      if (ds.includes('g')) return 'g';
      if (ds.includes('b')) return 'b';
      if (it.t === 'pick') return 'life';
      return it.k || it.t;
    }

    function isDunno(it, r) {
      if (it.t === 'know') return r === 3;
      if (it.t === 'quiz') return r === 2;
      if (it.t === 'pick' && !it.prof) return r === it.o.length || (it.none != null && r === it.none);
      return false;
    }

    function genreFactor() {
      const st = new Map();
      asked.forEach((a) => {
        const k = genreOf(a.item);
        const v = st.get(k) || { n: 0, d: 0 };
        v.n += 1;
        if (a.dunno) v.d += 1;
        st.set(k, v);
      });
      const f = new Map();
      st.forEach((v, k) => {
        if (v.n >= 3 && v.d / v.n >= 0.67) f.set(k, R.genreLo);
        else if (v.n >= 2 && v.d / v.n >= 0.5) f.set(k, R.genreMid);
      });
      return f;
    }

    const prior = M.prior;
    let post = TW.map((w) => prior.map((v) => v * w));

    const marg = () => {
      const m = new Array(N).fill(0);
      for (let t = 0; t < NT; t++) {
        const pt = post[t];
        for (let i = 0; i < N; i++) m[i] += pt[i];
      }
      return m;
    };

    let massCache = new Map();
    function massFor(mask) {
      let mm = massCache.get(mask);
      if (mm) return mm;
      mm = new Map();
      for (let t = 0; t < NT; t++) {
        const mv = t & mask;
        let a = mm.get(mv);
        if (!a) { a = new Array(N).fill(0); mm.set(mv, a); }
        const pt = post[t];
        for (let i = 0; i < N; i++) a[i] += pt[i];
      }
      massCache.set(mask, mm);
      return mm;
    }

    const asked = [];
    const used = new Set();
    let rewind = 0;

    function confOf(p) {
      let best = -1;
      let at = AGES[0];
      for (let i = 0; i < N; i++) {
        const m = (p[i - 1] || 0) + p[i] + (p[i + 1] || 0);
        if (m > best) { best = m; at = AGES[i]; }
      }
      return { age: at, conf: best };
    }

    function spread(p, mass) {
      let best = N;
      let lo = 0;
      let s = 0;
      for (let hi = 0; hi < N; hi++) {
        s += p[hi];
        while (lo < hi && s - p[lo] >= mass) { s -= p[lo]; lo++; }
        if (s >= mass) best = Math.min(best, hi - lo);
      }
      return best;
    }

    const CBIN = AGES.map((a) => Math.floor((a - AMIN) / R.coarseBin));
    const NCB = CBIN[N - 1] + 1;

    function infoGain(it, coarse, m) {
      const LI = L.get(it.id);
      const K = LI.byT[0][0].length;
      const J = [];
      for (let r = 0; r < K; r++) J.push(new Array(N).fill(0));
      const mm = massFor(LI.mask);
      LI.mvs.forEach((mv) => {
        const pt = mm.get(mv);
        const rt = LI.byMv.get(mv);
        for (let i = 0; i < N; i++) {
          const w = pt[i];
          if (w < 1e-14) continue;
          const row = rt[i];
          for (let r = 0; r < K; r++) J[r][i] += w * row[r];
        }
      });
      let after = 0;
      if (coarse) {
        const pb = new Array(NCB).fill(0);
        for (let i = 0; i < N; i++) pb[CBIN[i]] += m[i];
        for (let r = 0; r < K; r++) {
          let pr = 0;
          for (let i = 0; i < N; i++) pr += J[r][i];
          if (pr < 1e-9) continue;
          const qb = new Array(NCB).fill(0);
          for (let i = 0; i < N; i++) qb[CBIN[i]] += J[r][i] / pr;
          after += pr * H(qb);
        }
        return H(pb) - after;
      }
      for (let r = 0; r < K; r++) {
        let pr = 0;
        for (let i = 0; i < N; i++) pr += J[r][i];
        if (pr < 1e-9) continue;
        after += pr * H(J[r].map((v) => v / pr));
      }
      return H(m) - after;
    }

    function allowed(it, fine) {
      if (used.has(it.id)) return false;
      if (it.prof || it.home) return false;
      if (!fits(it)) return false;
      if (it.group && asked.some((a) => a.item.group === it.group)) return false;
      const n = asked.length;
      if ((it.t === 'anchor' || it.t === 'pick') && n < R.lateFrom) return false;
      if (it.t === 'anchor' && asked.filter((a) => a.item.t === 'anchor').length >= R.maxAnchor) return false;
      if (it.t === 'pick' && asked.filter((a) => a.item.t === 'pick').length >= (fine ? R.maxPick : R.maxPick1)) return false;
      if ((it.t === 'anchor' || it.t === 'pick') && n && (asked[n - 1].item.t === 'anchor' || asked[n - 1].item.t === 'pick')) return false;
      return true;
    }

    function next() {
      if (done()) return null;
      const n = asked.length;
      if (homeItem && !used.has(homeItem.id) && asked.filter((a) => a.dunno && a.pred < R.homeSurp && tvGame(a.item)).length >= R.homeAfter) return homeItem;
      if (profItem && !used.has(profItem.id) && asked.filter((a) => a.dunno).length >= R.profAfter) return profItem;
      tr = R.traits ? traitNow() : null;
      const gf = genreFactor();
      massCache = new Map();
      const m = marg();
      const fine = rewind <= 0 && spread(m, 0.8) <= R.coarse;
      const cands = items.filter((it) => allowed(it, fine));
      if (!cands.length) return null;

      const wantQuiz = R.quizEvery > 0 && n % R.quizEvery === R.quizEvery - 1;
      const lastKinds = asked.slice(-2).map((a) => a.item.k || a.item.t);
      const kindCounts = new Map();
      asked.forEach((a) => {
        const k = a.item.k || a.item.t;
        kindCounts.set(k, (kindCounts.get(k) || 0) + 1);
      });
      let bestRaw = 0;
      let scored = cands.map((it) => {
        const raw = infoGain(it, !fine, m);
        if (fine && raw > bestRaw) bestRaw = raw;
        let g = raw * (gf.get(genreOf(it)) || 1) * (jitter.get(it.id) || 1);
        if (recent.has(it.id)) g *= 0.55;
        const kind = it.k || it.t;
        if (lastKinds.length === 2 && lastKinds[0] === kind && lastKinds[1] === kind) g *= 0.35;
        else if (lastKinds[lastKinds.length - 1] === kind) g *= 0.65;
        const kc = kindCounts.get(kind) || 0;
        if (kc >= 3) g *= 0.75;
        if (fine && it.t === 'pick') g *= R.pickBoost;
        return { it, g };
      }).sort((a, b) => b.g - a.g);

      if (fine && n >= R.minQ && bestRaw < R.minGain) return null;
      if (wantQuiz) {
        const best = scored[0].g;
        const quiz = scored.filter((s) => s.it.t === 'quiz' && s.g >= best * R.quizMin);
        if (quiz.length) scored = quiz;
      } else {
        scored = scored.filter((s) => s.it.t !== 'quiz').concat(scored.filter((s) => s.it.t === 'quiz'));
      }

      const bestG = scored[0].g;
      const poolMax = n < 5 ? 18 : (n < 12 ? 12 : 8);
      const floor = n < 5 ? 0.55 : 0.62;
      const near = scored.filter((s) => s.g >= bestG * floor);
      const top = (near.length >= 4 ? near : scored).slice(0, poolMax);
      const ws = top.map((s) => Math.max(1e-9, s.g));
      let sum = 0;
      ws.forEach((w) => { sum += w; });
      let x = rand() * sum;
      for (let i = 0; i < top.length; i++) {
        x -= ws[i];
        if (x <= 0) return top[i].it;
      }
      return top[0].it;
    }

    function answer(it, r) {
      const rowsT = L.get(it.id).byT;
      const before = marg();
      const wasFine = rewind <= 0 && spread(before, 0.8) <= R.coarse;
      if (rewind > 0) rewind -= 1;

      let pred = 0;
      for (let t = 0; t < NT; t++) {
        const pt = post[t];
        const rt = rowsT[t];
        for (let i = 0; i < N; i++) pred += pt[i] * rt[i][r];
      }

      const rowsP = powRows(L.get(it.id));
      let tot = 0;
      post = post.map((pt, t) => pt.map((v, i) => {
        const x = v * rowsP[t][i][r];
        tot += x;
        return x;
      }));
      if (tot > 0) post = post.map((pt) => pt.map((v) => v / tot));

      let surprise = false;
      if (wasFine && R.surprise > 0 && pred < R.surprise && asked.filter((a) => a.surprise).length < R.maxRewind) {
        surprise = true;
        let s2 = 0;
        post = post.map((pt) => pt.map((v) => {
          const x = Math.pow(v, R.widen);
          s2 += x;
          return x;
        }));
        if (s2 > 0) post = post.map((pt) => pt.map((v) => v / s2));
        rewind = R.rewindQ;
      }

      used.add(it.id);
      asked.push({ item: it, r, pred, gain: H(before) - H(marg()), surprise, fine: wasFine, dunno: isDunno(it, r) });
      return state();
    }

    function done() {
      const n = asked.length;
      if (n >= R.maxQ) return true;
      if (n >= R.minQ && confOf(marg()).conf >= R.conf) return true;
      return false;
    }

    function state() {
      const c = confOf(marg());
      return { n: asked.length, conf: c.conf, guess: c.age, done: done() };
    }

    return {
      next,
      answer,
      done,
      state,
      get post() { return marg(); },
      get types() {
        const keys = ['retro', 'parent', 'game', 'girl', 'out', 'tv'];
        const o = {};
        keys.forEach((k, b) => {
          let s1 = 0;
          for (let t = 0; t < NT; t++) if ((t >> b) & 1) s1 += post[t].reduce((a, v) => a + v, 0);
          o[k] = s1;
        });
        return o;
      },
      get asked() { return asked.slice(); },
      items,
      likes: (id) => L.get(id),
    };
  }

  return {
    AMIN, AMAX, AGES, BIN, NBIN, binOf,
    create, model, knowProbs, quizProbs, anchorProbs, pickProbs, pickProbsTy,
    profProbs, homeProbs, changeOf, likeOf, arity, basePrior, STAGES,
  };
});
