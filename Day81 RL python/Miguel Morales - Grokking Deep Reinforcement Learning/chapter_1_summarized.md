# Chapter 1: Introduction to Deep Reinforcement Learning - အကျဉ်းချုပ်

## 1. Deep Reinforcement Learning (DRL) ဆိုတာဘာလဲ

Deep Reinforcement Learning ဆိုတာ **deep learning** နဲ့ **reinforcement learning** ကို ပေါင်းစပ်ထားတဲ့ နည်းပညာတစ်ခုဖြစ်ပါတယ်။ Neural network (deep learning) ရဲ့ function approximation စွမ်းရည်ကို အသုံးပြုပြီး ရှုပ်ထွေးတဲ့ sequential decision-making ပြဿနာတွေကို uncertainty အောက်မှာ ဖြေရှင်းတဲ့ နည်းလမ်းဖြစ်ပါတယ်။

DRL ရဲ့ စိန်ခေါ်မှုတွေကို feedback အမျိုးအစား (၃) ခုနဲ့ ရှင်းပြနိုင်ပါတယ်:

- **Sequential feedback** — Agent ရဲ့ action တွေက ချက်ချင်းရလဒ်သာမက ရေရှည်ရလဒ်ကိုပါ သက်ရောက်ပါတယ်။ ဒါကြောင့် immediate နဲ့ long-term goals ကို ချိန်ညှိရပါတယ်။
- **Evaluative feedback** — Agent ကို မှန်တဲ့အဖြေ မပေးဘဲ action ကောင်း/မကောင်းကိုပဲ ပြောပါတယ်။ ဒါကြောင့် exploration (စူးစမ်းမှု) နဲ့ exploitation (အသုံးချမှု) ကို balance ညှိရပါတယ်။
- **Sampled feedback** — Agent က ကြုံတွေ့ရတဲ့ experience အနည်းငယ်ကနေ generalize လုပ်ရပါတယ်။

---

## 2. Machine Learning ရဲ့ အမျိုးအစားများနှင့် DRL ရဲ့ နေရာ

Machine Learning (ML) ကို အဓိက (၃) မျိုး ခွဲခြားနိုင်ပါတယ်:

| အမျိုးအစား | ဖော်ပြချက် |
|---|---|
| **Supervised Learning (SL)** | Labeled data ကနေ သင်ယူခြင်း — $f: X \rightarrow Y$ |
| **Unsupervised Learning (UL)** | Label မပါတဲ့ data ထဲက pattern ရှာခြင်း |
| **Reinforcement Learning (RL)** | Environment နဲ့ interact လုပ်ပြီး reward ကနေ သင်ယူခြင်း |

Deep Learning ဆိုတာ multiple layers ပါတဲ့ neural network ကို အသုံးပြုတဲ့ ML approach ဖြစ်ပြီး ML ရဲ့ branch တစ်ခုခုမှာပဲ ကန့်သတ်ထားတာ မဟုတ်ပါဘူး။ ML community တစ်ခုလုံးကို တိုးတက်စေပါတယ်။

$$\text{DRL} = \text{Deep Learning} + \text{Reinforcement Learning}$$

---

## 3. Artificial Intelligence ရဲ့ သမိုင်းကြောင်း

### AI ရဲ့ အစပိုင်း
- **Alan Turing** (1930-1950s) — Turing Test ကို တီထွင်ခဲ့ပြီး machine intelligence ကို တိုင်းတာဖို့ စံသတ်မှတ်ခဲ့ပါတယ်။
- **John McCarthy** (1955) — "Artificial Intelligence" ဆိုတဲ့ အသုံးအနှုန်းကို စတင်သုံးစွဲခဲ့ပါတယ်။ 1956 မှာ ပထမဆုံး AI conference ကို ဦးဆောင်ကျင်းပခဲ့ပြီး Lisp programming language ကိုလည်း တီထွင်ခဲ့ပါတယ်။

### AI Winters (အေးခဲချိန်များ)
AI သမိုင်းတစ်လျှောက် **hype** (အလွန်အကျွံ မျှော်လင့်ချက်) နဲ့ **disillusion** (စိတ်ပျက်မှု) pattern ကို ထပ်ခါထပ်ခါ ကြုံတွေ့ခဲ့ရပါတယ်။ AI researcher တွေက လူသားနဲ့တူတဲ့ machine intelligence ကို နှစ်အနည်းအကျဉ်းအတွင်း ဖန်တီးနိုင်မယ်လို့ ကြွားဝါခဲ့ပေမယ့် အထမြောက်ခဲ့ခြင်း မရှိပါဘူး။ ဒါကြောင့် funding နဲ့ စိတ်ဝင်စားမှု ကျဆင်းတဲ့ "AI winter" ကာလတွေ ဖြစ်ပေါ်ခဲ့ပါတယ်။

### AI ရဲ့ လက်ရှိအခြေအနေ
ယနေ့ခေတ်မှာ Google, Facebook, Microsoft, Amazon, Apple စတဲ့ ကမ္ဘာ့အကြီးဆုံးကုမ္ပဏီတွေက AI research ကို အကြီးအကျယ် ရင်းနှီးမြှုပ်နှံနေပါတယ်။ Computing power၊ big data၊ top researcher team တွေ ရှိနေတဲ့အတွက် AI research ပိုမိုတည်ငြိမ်ပြီး ထိရောက်လာနေပါတယ်။

---

## 4. Deep Reinforcement Learning ရဲ့ တိုးတက်မှုများ

### အဓိက မှတ်တိုင်များ

| ခုနှစ် | အောင်မြင်မှု |
|---|---|
| **1990s** | **TD-Gammon** — Gerald Tesauro က ANN သုံးပြီး backgammon ကစားတဲ့ program ဖန်တီးခဲ့ |
| **2004** | Andrew Ng — Inverse RL သုံးပြီး autonomous helicopter အလိုအလျောက် ပျံသန်းခြင်း |
| **2013/2015** | **DQN** (Mnih et al.) — Atari games ကို raw pixel ကနေ သင်ယူကစားခြင်း။ CNN + single hyperparameter set သုံးပြီး 49 games ထဲက 22 games မှာ professional player ထက် သာခဲ့ |
| **2014** | **DPG** (Silver et al.) — Deterministic Policy Gradient |
| **2015** | **DDPG** (Lillicrap et al.) — Deep Deterministic Policy Gradient |
| **2016** | **TRPO**, **GAE**, **GPS**, **AlphaGo** |
| **2017** | **AlphaZero**, **PPO**, **A3C**, **A2C**, **Rainbow** စတဲ့ algorithms များ |
| **2019** | **AlphaStar** — StarCraft II မှာ professional player တွေကို နိုင်ခဲ့ |
| **2019** | **OpenAI Five** — Dota 2 world champions ကို ပထမဆုံး AI အနေနဲ့ နိုင်ခဲ့ |

ဆယ်စုနှစ် (၂) ခုအတွင်းမှာ:

$$\text{Backgammon} (10^{20} \text{ states}) \xrightarrow{} \text{Go} (10^{170} \text{ states}) \xrightarrow{} \text{StarCraft II} (10^{270} \text{ states})$$

ဒီလို exponential တိုးတက်မှုဟာ DRL field ရဲ့ အလားအလာကို ရှင်းရှင်းလင်းလင်း ပြသနေပါတယ်။

---

## 5. DRL ရဲ့ အားသာချက်များနှင့် အားနည်းချက်များ

### အားသာချက်များ (Strengths)
- **Well-defined single tasks** မှာ အထူးကောင်းမွန် — RL agent တွေက specific task တစ်ခုကို master လုပ်ဖို့ အထူးကောင်းပါတယ်။
- **Raw sensory input** ကနေ တိုက်ရိုက် skill သင်ယူနိုင် — Deep learning ရဲ့ generalization technique တွေကို အသုံးချနိုင်ပါတယ်။
- DL advancement တိုင်းက DRL research path အသစ်တွေကို ဖွင့်ပေးပါတယ်။

### အားနည်းချက်များ (Weaknesses)

1. **Sample Inefficiency** — Agent တွေက ကောင်းမွန်တဲ့ policy သင်ယူဖို့ sample သန်းချီ လိုအပ်ပါတယ်။ လူသားတွေက interaction အနည်းငယ်နဲ့ သင်ယူနိုင်ပေမယ့် DRL agent တွေက မရနိုင်သေးပါ။

2. **Reward Function Design** — Task တစ်ခုအတွက် reward function ကို ဘယ်လိုသတ်မှတ်မလဲဆိုတာ စိန်ခေါ်မှုဖြစ်ပါတယ်။
   - Dense reward $\rightarrow$ သင်ယူမှုမြန် ဒါပေမယ့် solution မထူးဆန်း
   - Sparse reward $\rightarrow$ solution ပိုထူးဆန်း ဒါပေမယ့် သင်ယူရခက်

3. **Exploration Risk** — Agent တွေက mistakes လုပ်ရမှာဖြစ်ပြီး real-world application မှာ ဒါက ပြဿနာဖြစ်နိုင်ပါတယ် (ဥပမာ — self-driving car agent တစ်ခုက crash မလုပ်ဖို့ crash လုပ်ပြီးမှ သင်ယူရမှာလား?)

### အသုံးဝင်တဲ့ ချဉ်းကပ်နည်းများ
- **Transfer Learning** — Task တစ်ခုကနေ ရရှိတဲ့ knowledge ကို task အသစ်မှာ reuse လုပ်ခြင်း
- **Hierarchical RL** — Action hierarchy ဖန်တီးပြီး reusable sub-skills သင်ယူခြင်း
- **Intrinsic Motivation** — Curiosity-driven exploration ဖြင့် sparse reward environment မှာ performance တိုးတက်စေခြင်း

---

## 6. DRL Algorithm အမျိုးအစားများ

စာအုပ်ထဲမှာ ဖော်ပြထားတဲ့ DRL algorithmic approaches တွေကို အောက်ပါအတိုင်း နှိုင်းယှဉ်နိုင်ပါတယ်:

$$\underbrace{\text{Derivative-free}}_{\text{Sample efficient ↑}} \leftarrow \underbrace{\text{Policy-based}}_{} \leftarrow \underbrace{\text{Actor-critic}}_{} \leftarrow \underbrace{\text{Value-based}}_{} \leftarrow \underbrace{\text{Model-based}}_{\text{Sample efficient ↓}}$$

| Feature | Derivative-free | Policy-based | Actor-critic | Value-based | Model-based |
|---|---|---|---|---|---|
| Sample Efficiency | နိမ့် | နိမ့် | အလယ်အလတ် | မြင့် | အမြင့်ဆုံး |
| Computation Cost | နိမ့် | အလယ် | အလယ် | မြင့် | အမြင့်ဆုံး |
| Direct Learning | တိုက်ရိုက်ဆုံး | တိုက်ရိုက် | အလယ် | သွယ်ဝိုက် | သွယ်ဝိုက်ဆုံး |

---

## 7. စာအုပ်ရဲ့ ဖွဲ့စည်းပုံ

စာအုပ်ကို အပိုင်းကြီး (၂) ပိုင်း ခွဲထားပါတယ်:

### Part 1: Tabular RL (Chapters 3–7)
Neural network မလိုအပ်တဲ့ RL ပြဿနာများ:
- **Chapter 3** — Sequential aspect + temporal credit assignment
- **Chapter 4** — Evaluative feedback + exploration vs exploitation
- **Chapter 5** — Fixed behavior ရဲ့ result ခန့်မှန်းခြင်း (Policy Evaluation)
- **Chapter 6** — Behavior တိုးတက်စေခြင်း (Policy Improvement)
- **Chapter 7** — RL ကို ပိုထိရောက်အောင်လုပ်တဲ့ technique များ

### Part 2: Deep RL (Chapters 8–12)
Core DRL algorithms:
- **Chapters 8–10** — Value-based DRL (DQN, DDQN, PER, Rainbow, etc.)
- **Chapter 11** — Policy-based DRL + Actor-Critic
- **Chapter 12** — DPG, SAC, PPO methods

---

## 8. Development Environment

- **Docker** — ကြိုတင်ပြင်ဆင်ထားတဲ့ Docker image နဲ့ Jupyter Notebooks
- **Python** — NumPy နဲ့ **PyTorch** ကို အဓိကသုံး (PyTorch က "Pythonic" ဖြစ်ပြီး research/teaching အတွက် ပိုသင့်တော်)
- **OpenAI Gym** — RL agent training အတွက် environment library
- **GPU** မဖြစ်မနေ မလို — DRL architecture တွေက DL model တွေလောက် computation မများ။ CPU core အရေအတွက်ကသာ bottleneck ဖြစ်တတ်ပါတယ်။

---

## 9. နိဂုံးချုပ်

> Deep Reinforcement Learning ဆိုတာ **sequential**, **evaluative**, **sampled** feedback (၃) မျိုးကနေ တစ်ပြိုင်နက်တည်း သင်ယူရတဲ့ စိန်ခေါ်မှုကြီးတစ်ခုဖြစ်ပါတယ်။

DRL ရဲ့ အဓိက လိုအပ်ချက်များ:
1. **Sample complexity** ပိုကောင်းတဲ့ algorithms
2. **Exploration strategies** ပိုကောင်းတဲ့ methods
3. **Safe algorithms** — လုံခြုံစိတ်ချရတဲ့ algorithm များ

DRL field ရဲ့ အနာဂတ်က တောက်ပနေပြီး ဒီနည်းပညာရဲ့ အလားအလာက အကန့်အသတ်မရှိပါဘူး။ Industrial revolution, digital revolution တွေလိုပဲ AI revolution ကလည်း ရေတိုမှာ အခက်အခဲရှိနိုင်ပေမယ့် ရေရှည်မှာ လူသားအားလုံးကို အကျိုးပြုမှာ ဖြစ်ပါတယ်။
