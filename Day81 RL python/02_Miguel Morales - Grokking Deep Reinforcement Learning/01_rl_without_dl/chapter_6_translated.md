# အခန်း ၆ — Agent များ၏ အပြုအမူကို တိုးတက်စေခြင်း (Improving Agents' Behaviors)

> *"ပန်းတိုင်ကို မရောက်နိုင်တော့ဘူးဆိုတာ ထင်ရှားနေတဲ့အခါ၊ ပန်းတိုင်ကို မပြင်ပါနဲ့၊ လုပ်ဆောင်ရမယ့် အဆင့်တွေကိုပဲ ပြင်ဆင်ပါ။"*
> — ကွန်ဖျူးရှပ် (Confucius)

## ဒီအခန်းမှာ သင်ယူရမည့်အချက်များ

- Feedback က sequential ရော evaluative ပါ တစ်ပြိုင်နက်ဖြစ်နေတဲ့ RL ပတ်ဝန်းကျင်တွေမှာ policy တွေကို ပိုကောင်းအောင် improve လုပ်နည်းကို သင်ယူရမည်။
- Transition function နှင့် reward function မသိရတဲ့ RL ပတ်ဝန်းကျင်တွေမှာ optimal policy ကို ရှာဖွေနိုင်သော algorithm များကို ဖန်တီးရမည်။
- Agent များကို random behavior ကနေ optimal behavior ဆီ ရောက်အောင် code ရေးပြီး environment အမျိုးမျိုးမှာ train လုပ်ရမည်။

---

## ၆.၁ — နိဒါန်း

ဒီအခန်းမတိုင်ခင်အထိ၊ reinforcement learning agent တစ်ခုက ရင်ဆိုင်ရမယ့် feedback အမျိုးအစား သုံးခု — **sequential**, **evaluative**, နှင့် **sampled** — ကို သီးခြားစီရောပါ လေ့လာခဲ့ပြီးပြီ။

- **Chapter 2** — Sequential decision-making ပြဿနာတွေကို **Markov Decision Processes (MDPs)** ဖြင့် ကိုယ်စားပြုနည်း
- **Chapter 3** — MDP တွေကနေ policy ထုတ်ယူနည်း (Policy Iteration, Value Iteration)
- **Chapter 4** — MDP မသိဘဲ **Multi-Armed Bandits** ပြဿနာတွေ ဖြေရှင်းနည်း (exploration strategies)
- **Chapter 5** — Sequential + evaluative feedback နှစ်မျိုးလုံး ရှိတဲ့ **prediction problem** (value function estimation)

ဒီအခန်းမှာတော့ **control problem** ကို ဖြေရှင်းမယ်။ ပြောင်းလဲမှု အဓိက ၂ ခု ရှိတယ်:

1. **State-value function** $V(s)$ အစား **action-value function** $Q(s, a)$ ကို estimate လုပ်မယ်။ $Q$-function ကို သုံးခြင်းဖြင့် MDP မလိုဘဲ action တွေရဲ့ value ကို တိုက်ရိုက်မြင်နိုင်မယ်။
2. $Q$-value estimate တွေကို ရရှိပြီးနောက်၊ policy တွေကို **improve** လုပ်မယ်။ ဒါက policy iteration algorithm မှာ ကျွန်တော်တို့လုပ်ခဲ့တဲ့ pattern နဲ့ တူပါတယ် — evaluate, improve, ထပ် evaluate, ထပ် improve ... ဒီ pattern ကို **Generalized Policy Iteration (GPI)** လို့ ခေါ်ပါတယ်။

**ဒီအခန်းရဲ့ outline:**

1. GPI architecture ကို ချဲ့ထွင်ရှင်းပြမယ်
2. Control problem ဖြေရှင်းတဲ့ agent အမျိုးမျိုးကို သင်ယူမယ် — MC Control, SARSA, Q-learning
3. Learning နဲ့ behavior ကို decouple လုပ်တဲ့ agent မျိုးကို လေ့လာမယ် — Q-learning, Double Q-learning
4. Trial-and-error learning ဖြင့် optimal policy ကို ရှာဖွေမယ်

---

## ၆.၂ — Reinforcement Learning Agents များ၏ ခန္ဓာဗေဒ (The Anatomy of RL Agents)

RL agent တိုင်းက အောက်ပါ pattern ကို အကြမ်းအားဖြင့် လိုက်နာပါတယ်:

### ၆.၂.၁ — Agent တိုင်း experience sample များ စုဆောင်းသည်

Agent တိုင်းက environment နဲ့ interact လုပ်ခြင်း (ဒါမှမဟုတ် learned model ကို query လုပ်ခြင်း) ဖြင့် data စုဆောင်းပါတယ်။ Agent သင်ယူနေစဉ်မှာပဲ data ထွက်ပေါ်လာပါတယ်။

### ၆.၂.၂ — Agent အများစုက တစ်ခုခု estimate လုပ်သည်

Data ကို ရရှိပြီးတဲ့အခါ agent က အမျိုးမျိုးသော အရာတွေကို estimate လုပ်နိုင်ပါတယ်:

- **Value functions** — MC target, TD target, n-step target, λ-return target စသည်ဖြင့် estimate ပုံ အမျိုးမျိုးရှိတယ်
- **Environment models** — Model-based RL agent တွေက transition function နဲ့ reward function ကို သင်ယူတယ်
- **Policies** — Policy gradient နဲ့ actor-critic method တွေက state ကို ယူပြီး action probability distribution ထုတ်ပေးတဲ့ function ကို approximate လုပ်တယ်
- **Multiple things** — Agent တစ်ခုက value function ရော policy ရော model ရော တစ်ပြိုင်နက် estimate လုပ်နိုင်တယ်

### ၆.၂.၃ — Agent အများစုက policy ကို improve လုပ်သည်

Agent တိုင်းရဲ့ အဆုံးစွန်ပန်းတိုင်က policy ကို ပိုကောင်းအောင် လုပ်ခြင်းပါပဲ။

- Agent က **value function** ကို estimate လုပ်ထားရင် → value function ထဲက implicitly encode ထားတဲ့ **target policy** ကို improve လုပ်တယ်
- Agent က **explicit policy** (policy gradient) ကို estimate လုပ်ထားရင် → actual returns ဒါမှမဟုတ် estimated value functions ကို သုံးပြီး policy ကို တိုက်ရိုက် improve လုပ်တယ်
- **Model-based RL** မှာ → learned model ကို သုံးပြီး action sequence plan ဆွဲတယ်၊ ဒါမှမဟုတ် value function/policy ကို improve လုပ်တယ်

---

### RL သဘာဝစကား — Greedy vs. Epsilon-greedy vs. Optimal Policy

| အသုံးအနှုန်း | အဓိပ္ပါယ် |
|---|---|
| **Greedy policy** | State တိုင်းကနေ **ယုံကြည်ထားတဲ့** highest expected return ပေးမယ့် action ကို အမြဲရွေးတဲ့ policy။ ဘယ် value function ကို ရည်ညွှန်းလဲ ဆိုတာ အရေးကြီးတယ်။ Random value function အပေါ် greedy ဖြစ်ရင် policy ညံ့ပါလိမ့်မယ်။ |
| **Epsilon-greedy policy** | State တိုင်းကနေ **များသောအားဖြင့်** highest expected return ပေးမယ့် action ကို ရွေးတဲ့ policy။ $\epsilon$ probability ဖြင့် random action ယူပြီး explore လုပ်တယ်။ |
| **Optimal policy** | State တိုင်းကနေ **အမှန်တကယ်** highest expected return ပေးတဲ့ action ကို အမြဲရွေးတဲ့ policy။ Optimal policy ဟာ **optimal value function** အပေါ် greedy ဖြစ်ရမယ်။ |

---

### RL သဘာဝစကား — Non-interactive vs. Interactive Learning

| အမျိုးအစား | ရှင်းလင်းချက် |
|---|---|
| **Non-interactive** | Environment နဲ့ interact မလုပ်ဘဲ ကြိုတင်စုဆောင်းထားတဲ့ data ကနေ သင်ယူတယ်။ ဥပမာ — inverse RL, apprenticeship learning, behavioral cloning |
| **Interactive** | Learning နဲ့ interaction ကို ယှက်နွယ်ပြီး လုပ်ဆောင်တယ်။ Learner ကိုယ်တိုင် data-gathering process ကို ထိန်းချုပ်တယ်။ |

---

### Refresh — Rewards, Returns, နှင့် Value Functions

ဒီအခန်းကို ဆက်မသွားခင် ဒါတွေကို ပြန်သတိရဖို့ လိုအပ်ပါတယ်:

**Reward** — Transition တစ်ခု၏ ကောင်းမွန်မှုကို ညွှန်ပြတဲ့ numeric signal တစ်ခု။ Agent က state $S_t$ ကို observe လုပ်ပြီး action $A_t$ ယူတယ်၊ environment ပြောင်းပြီး reward $R_{t+1}$ ပေးတယ်၊ next state $S_{t+1}$ ထုတ်ပေးတယ်။

**Return** — Episode တစ်ခုတွင်း ရရှိတဲ့ reward များ၏ discounted ပေါင်းလဒ်:

$$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$$

**Value function** — State (ဒါမှမဟုတ် state-action pair) တစ်ခုမှ ရနိုင်သော **expected return**:

$$V^\pi(s) = \mathbb{E}_\pi [G_t \mid S_t = s]$$

$$Q^\pi(s, a) = \mathbb{E}_\pi [G_t \mid S_t = s, A_t = a]$$

---

### Refresh — Monte Carlo vs. Temporal-Difference Targets

Learning method အားလုံး အောက်ပါ general equation ကို follow လုပ်ပါတယ်:

$$\text{estimate} \leftarrow \text{estimate} + \alpha \times (\text{target} - \text{estimate})$$

**Monte Carlo target** — Actual return ကို target အဖြစ် သုံးတယ်:

$$\text{MC target} = G_t = R_{t+1} + \gamma R_{t+2} + \cdots + \gamma^{T-t-1} R_T$$

> MC target ကတော့ actual return ကိုပဲ သုံးတယ်။ Episode ပြီးမှ compute လုပ်လို့ရတယ်။ Unbiased ပေမယ့် high variance ဖြစ်တတ်တယ်။

**TD target** — Estimated return ကို target အဖြစ် သုံးတယ် (bootstrapping):

$$\text{TD target} = R_{t+1} + \gamma V(S_{t+1})$$

> TD target ကတော့ reward တစ်ခု ယူပြီး next state ရဲ့ estimated value ကို ပေါင်းထည့်တယ်။ "Guess ကနေ guess ကို သင်ယူခြင်း" ပါပဲ။ Low variance ပေမယ့် biased ဖြစ်နိုင်တယ်။

---

## ၆.၃ — Generalized Policy Iteration (GPI)

RL algorithm အများစုရဲ့ architecture ကို နားလည်ဖို့ အသုံးအဝင်ဆုံး pattern က **Generalized Policy Iteration (GPI)** ပါ။ GPI ဆိုတာ **policy evaluation** နှင့် **policy improvement** ၏ စဉ်ဆက်မပြတ် interaction က policy တွေကို optimality ဆီ တွန်းပို့တယ်ဆိုတဲ့ ယေဘုယျ အယူအဆ ဖြစ်ပါတယ်။

### GPI ၏ Pattern

Policy iteration algorithm မှာ process ၂ ခု ရှိခဲ့တယ်:

1. **Policy Evaluation** — Policy တစ်ခုယူပြီး အဲ့ policy ရဲ့ value function ကို estimate လုပ်တယ်
2. **Policy Improvement** — Value function ကို အသုံးပြုပြီး ပိုကောင်းတဲ့ policy ယူတယ်

ဒီ process ၂ ခု stabilize ဖြစ်ပြီး ပြောင်းလဲမှု မရှိတော့တဲ့အခါ (**policy evaluation** ကလည်း value function ကို ပြောင်းမပေး၊ **policy improvement** ကလည်း policy ကို ပြောင်းမပေးတော့တဲ့အခါ)、policy နဲ့ value function က **optimal** ဖြစ်သွားပါတယ်။

**Value iteration** ကလည်း GPI pattern ထဲ ကျပါတယ် — policy evaluation phase ကို **truncate** လုပ်ပြီး (iteration ၁ ခုပဲ) policy improvement ချက်ချင်းလုပ်တယ်။ Evaluation phase ကို fully converge မလုပ်ဘဲ truncate လုပ်ရင်တောင် GPI pattern ကြောင့် optimal policy ကို ရနိုင်ပါတယ်။

### GPI ၏ Key Insight

- **Policy evaluation** → value function ကို current policy နဲ့ consistent ဖြစ်အောင် လုပ်ပေးတယ်
- **Policy improvement** → ဒီ consistency ကို ဖျက်ပေမယ့်、ပိုကောင်းတဲ့ policy ထုတ်ပေးတယ်
- ဒီ process ၂ ခု interact ဖြစ်ခြင်းဖြင့် ပိုပိုကောင်းတဲ့ policy တွေကို iteratively ထုတ်ပေးပြီး optimal policy ဆီ converge ဖြစ်သွားတယ်

$$\text{Policy Evaluation} \xrightarrow{\text{V ကို } \pi \text{ နဲ့ consistent ဖြစ်အောင်}} \text{Policy Improvement} \xrightarrow{\pi \text{ ကို V အပေါ် greedy ဖြစ်အောင်}} \text{Better Policy}$$

### DP Environment ရှိတုန်းက vs. RL Environment

DP (Chapter 3) မှာတော့ MDP ရှိတဲ့အတွက် policy ကို **completely greedy** ဖြစ်အောင် လုပ်နိုင်ခဲ့တယ်။ RL (ယခု) မှာတော့ MDP မရှိတဲ့အတွက် sample-based evaluation ကို သုံးရပြီး **exploration** ကို ထည့်ရမယ်။ Policy ကို totally greedy မဖြစ်အောင်、**ε-greedy** (greedier, but still exploring) ဖြစ်အောင်ပဲ လုပ်ရမယ်။

> **Miguel ၏ Analogy:** GPI ဟာ **critic (ဝေဖန်သူ)** နှင့် **performer (စွမ်းဆောင်သူ)** ၏ ထာဝရ dance နဲ့ တူပါတယ်။ Policy evaluation က feedback ပေးတယ်၊ policy improvement က ဒီ feedback ကို သုံးပြီး ပိုကောင်းအောင် လုပ်တယ်။ Benjamin Franklin ပြောခဲ့သလို "Critics are our friends, they show us our faults." GPI pattern ကို လူ့ဘဝမှာလည်း သုံးနိုင်ပါတယ် — data-driven decisions ဆိုတာ ကောင်းမွန်တဲ့ policy evaluation process ကို အသုံးပြုပြီး policy improvement process ကို result ကောင်းရအောင် လုပ်ခြင်းပါပဲ။

---

### RL အသုံးအနှုန်း — Prediction vs. Control Problem

| အသုံးအနှုန်း | အဓိပ္ပါယ် |
|---|---|
| **Prediction problem** | Policy တစ်ခု၏ value function ကို estimate လုပ်ခြင်း။ Returns ကို predict တတ်အောင် သင်ယူခြင်း။ |
| **Control problem** | Optimal policy ကို ရှာဖွေခြင်း။ GPI pattern ဖြင့် policy evaluation + improvement ကို combine လုပ်ပြီး ဖြေရှင်းတယ်။ |
| **Policy evaluation** | Prediction problem ကို solve လုပ်သော algorithms (MC, TD, n-step, TD(λ), etc.) |
| **Policy improvement** | Value function ကို ကြည့်ပြီး policy ကို greedier ဖြစ်အောင် update လုပ်ခြင်း။ တစ်ခုတည်းနဲ့ control problem ကို solve မရ။ |

---

## ၆.၄ — Policy များ၏ အပြုအမူကို ပိုကောင်းအောင် သင်ယူခြင်း (Learning to Improve Policies of Behavior)

Chapter 5 မှာ **prediction problem** ကို ဖြေရှင်းခဲ့တယ် — agent ကို policy တစ်ခု၏ value function ကို accurately estimate လုပ်တတ်အောင် လုပ်ခဲ့တယ်။ ပေမယ့် value function estimate လုပ်တတ်ရုံနဲ့ task ကို ပိုကောင်းအောင် ဖြေရှင်းနိုင်မှာ မဟုတ်ဘူး။

ဒီ section မှာ **control problem** ကို ဖြေရှင်းမယ် — agent ကို policy optimize လုပ်တတ်အောင် လုပ်မယ်။ Agent က **random policy** ကနေ စပြီး **trial-and-error learning** ဖြင့် **optimal policy** ဆီ ရောက်သွားမယ်။

GPI pattern ကို leverage လုပ်မယ့် သဘော:
- **Policy evaluation phase** → Chapter 5 က algorithms (MC, TD) ကို ရွေးသုံးမယ်
- **Policy improvement phase** → Chapter 4 က exploration strategies (ε-greedy) ကို ရွေးသုံးမယ်

---

### ၆.၄.၁ — Slippery Walk Seven (SWS) Environment

ဒီအခန်းရဲ့ experiment တွေအတွက် **Slippery Walk Seven (SWS)** environment ကို အသုံးပြုပါတယ်။

**SWS ၏ ဖွဲ့စည်းပုံ:**

```
[☠️ Terminal 0] — [S1] — [S2] — [S3] — [S4] — [S5] — [S6] — [S7] — [🏆 Terminal 8, +1]
```

- **Non-terminal states:** 7 ခု (states 1–7)
- **Terminal states:** State 0 (ဘယ်ဘက်, reward = 0) နှင့် State 8 (ညာဘက်, reward = +1)
- **Actions:** Left (0), Right (1)
- **Slippery (ချော်တတ်):**
  - 50% — ရည်ရွယ်ချက်အတိုင်း သွားရောက်
  - 33.3% — နေရာမှာပဲ ရပ်နေ
  - 16.7% — ဆန့်ကျင်ဘက်ကို သွားရောက်

> **အရေးကြီးချက်:** Agent က state ID နံပါတ်တွေ (0, 1, 2, ...) နဲ့ action နံပါတ်တွေ (0, 1) ကိုသာ မြင်ရပါတယ်။ State 3 က အလယ်မှာရှိတယ်ဆိုတာ မသိပါ။ Action 0 က ဘယ်ကိုသွားတယ်ဆိုတာလည်း မသိပါ။ Environment ၏ transition probabilities ကို ကျွန်တော်ကသာ didactic ရည်ရွယ်ချက်ဖြင့် ပြနေတာ ဖြစ်ပါတယ်။

<!-- 
📖 စာအုပ်ပုံ - SWS Environment MDP diagram (p.177)
Slippery Walk Seven MDP: States 0-8, transition probabilities for each action
- 50% intended direction
- 33.3% stay put  
- 16.7% opposite direction
-->

---

## ၆.၅ — Monte Carlo Control: Episode တိုင်းပြီးနောက် Policy ကို Improve လုပ်ခြင်း

MC prediction ကို policy evaluation အတွက် ယူသုံးပြီး control method ဖန်တီးကြည့်ကြမယ်။ ပေမယ့် prediction ကိုပဲ policy iteration algorithm ရဲ့ policy improvement (fully greedy) နဲ့ ပေါင်းလိုက်ရုံနဲ့ optimal policy ကို ရမလား? ဟင့်အင်း — **ပြောင်းလဲမှု ၂ ခု** လိုအပ်ပါတယ်:

### ပြောင်းလဲမှု ၁: $V(s)$ အစား $Q(s,a)$ estimate လုပ်ရမည်

$V$-function ကို MDP မရှိဘဲ ကြည့်ရင်、ဘယ် action ကို ယူရမလဲ ဆိုတာ ဆုံးဖြတ်လို့ မရပါ။

> ဥပမာ — States 2 ခုရှိပြီး $V(s_1) = 0.25$, $V(s_2) = 0.5$ ပဲ သိရင်၊ best action ဘာလဲ ဆိုတာ ဘယ်လို ပြောနိုင်မလဲ? Action Left က state $s_2$ ကို 70% ပေးတယ်ဆိုတာ MDP မှာမှ ပါတယ်။ Agent က MDP ကို မမြင်ရတဲ့အတွက် $Q(s,a)$ ကိုပဲ estimate လုပ်ဖို့ လိုပါတယ်။

### ပြောင်းလဲမှု ၂: Agent က explore လုပ်ရမည်

Greedy policy ကိုပဲ follow ခဲ့ရင်、ပိုကောင်းတဲ့ action ကို ဘယ်တော့မှ ရှာဖွေမတွေ့ရ။

> ဥပမာ — Agent က deterministic policy ဖြင့် ဘယ်ကိုသာ သွားနေတယ်ဆိုရင်、ညာဘက် action ရဲ့ value ကို ဘယ်တော့မှ estimate မလုပ်ရ။ ဒါကြောင့် explore ဖို့ လိုပါတယ်။

### MC Control Algorithm

- **Policy evaluation:** First-visit Monte Carlo prediction ($Q$-function estimate)
- **Policy improvement:** Decaying ε-greedy action-selection strategy
- Value iteration ကဲ့သို့ policy evaluation ကို **truncate** — episode ၁ ခုပြီးတိုင်း estimate + improve

### MC Control Update Rule

$$Q(s, a) \leftarrow Q(s, a) + \alpha \Big[ G_t - Q(s, a) \Big]$$

$G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$ (time step $t$ ကနေ ရရှိတဲ့ full return)

---

### Python Code — Decay Schedule

```python
def decay_schedule(init_value, min_value, decay_ratio, max_steps,
                   log_start=-2, log_base=10):
    """
    Alpha နှင့် epsilon များအတွက် exponentially decaying schedule ဖန်တီးခြင်း။
    init_value ကနေ min_value ဆီ decay_ratio * max_steps အတွင်း decay ဖြစ်ပြီး
    ကျန်တဲ့ steps မှာ min_value ကို ဆက်သုံးတယ်။
    """
    decay_steps = int(max_steps * decay_ratio)
    rem_steps = max_steps - decay_steps
    
    # Log space ကနေ decaying values တွက်
    values = np.logspace(log_start, 0, decay_steps,
                         base=log_base, endpoint=True)[::-1]
    
    # 0–1 range ကို normalize
    values = (values - values.min()) / (values.max() - values.min())
    
    # init_value – min_value range ကို linear transform
    values = (init_value - min_value) * values + min_value
    
    # ကျန်တဲ့ steps မှာ min_value ကို pad
    values = np.pad(values, (0, rem_steps), 'edge')
    return values
```

---

### Python Code — Exploratory Trajectory Generation

```python
def generate_trajectory(select_action, Q, epsilon, env, max_steps=200):
    """
    ε-greedy policy ဖြင့် environment မှာ episode တစ်ခုလုံး interact ပြီး
    trajectory (state, action, reward, ...) ကို ပြန်ပေးတယ်။
    """
    done, trajectory = False, []
    while not done:
        state = env.reset()
        for t in count():
            action = select_action(state, Q, epsilon)
            next_state, reward, done, _ = env.step(action)
            experience = (state, action, reward, next_state, done)
            trajectory.append(experience)
            if done or t >= max_steps - 1:
                break
            state = next_state
    return np.array(trajectory)
```

---

### Python Code — MC Control (အပြည့်အစုံ)

```python
def mc_control(env, gamma=1.0, init_alpha=0.5, min_alpha=0.01,
               alpha_decay_ratio=0.5, init_epsilon=1.0, min_epsilon=0.1,
               epsilon_decay_ratio=0.9, n_episodes=3000,
               max_steps=200, first_visit=True):
    """
    First-Visit Monte Carlo Control — episode ပြီးတိုင်း Q-function ကို update
    လုပ်ပြီး policy ကို ε-greedy ဖြင့် improve လုပ်တယ်။
    """
    nS, nA = env.observation_space.n, env.action_space.n

    # Discount factors ကို ကြိုတင်တွက် (max_steps = trajectory ရဲ့ max length)
    discounts = np.logspace(0, max_steps, num=max_steps, base=gamma, endpoint=False)

    # Alpha (learning rate) decaying schedule
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)

    # Epsilon decaying schedule — episode တိုင်း epsilon ကျသွားမယ်
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    # Q-function ကို zero ဖြင့် initialize
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    # ε-greedy action selection
    select_action = lambda state, Q, epsilon: \
        np.argmax(Q[state]) if np.random.random() > epsilon \
        else np.random.randint(len(Q[state]))

    for e in tqdm(range(n_episodes), leave=False):
        # Episode တစ်ခု generate
        trajectory = generate_trajectory(select_action, Q, epsilons[e], env, max_steps)

        # State-action pair visits ကို track
        visited = np.zeros((nS, nA), dtype=np.bool_)

        for t, (state, action, reward, _, _) in enumerate(trajectory):
            # First-visit check
            if visited[state][action] and first_visit:
                continue
            visited[state][action] = True

            # Return G_t ကို တွက် (discounted sum of future rewards)
            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps] * trajectory[t:, 2])

            # Q-function update
            Q[state][action] = Q[state][action] + alphas[e] * (G - Q[state][action])

        # Analysis အတွက် save
        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    # Final V-function နဲ့ greedy policy ထုတ်ယူ
    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, Q_track, pi_track
```

> **MC Control** သည် **offline method** (episode-to-episode) ဖြစ်တယ် — episode ပြီးမှသာ update လုပ်နိုင်တယ်။ Variance မြင့်ပေမယ့် bias နည်းတယ်။

---

## ၆.၆ — SARSA: Step တိုင်းပြီးနောက် Policy ကို Improve လုပ်ခြင်း

### MC ၏ အားနည်းချက်

MC methods ဟာ **offline** (episode-to-episode) ဖြစ်တဲ့အတွက် terminal state ကို ရောက်မှသာ value function estimate ကို update လုပ်နိုင်တယ်။ ဒါက episode ရှည်တဲ့ (ဒါမှမဟုတ် continuing) environments မှာ ပြဿနာ ဖြစ်တယ်။

### Solution — TD Prediction ကို အစားထိုးသုံးခြင်း

MC prediction အစား **TD prediction** ကို policy evaluation phase မှာ သုံးရုံပါပဲ — ဒါကို **SARSA** algorithm လို့ ခေါ်ပါတယ်။

### SARSA Update Equation

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \Big[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \Big]$$

- $A_{t+1}$ — agent က **အမှန်တကယ်ယူမယ့်** next action (ε-greedy ကနေ select)
- **TD target:** $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$
- **TD error:** $\text{target} - \text{estimate}$

> **SARSA** လို့ ဘာကြောင့် ခေါ်လဲ? — **(S**tate, **A**ction, **R**eward, next **S**tate, next **A**ction) — ဒီ tuple ၅ ခု ကို update လုပ်ဖို့ လိုအပ်တဲ့အတွက် ဖြစ်ပါတယ်။

### Planning Methods နှင့် Control Methods နှိုင်းယှဥ်ခြင်း

<!-- 📖 စာအုပ်ပုံ - Comparison diagram (p.183)
Policy Iteration: Full iterative policy evaluation + greedy improvement
Value Iteration: Truncated iterative evaluation + greedy improvement  
MC Control: Truncated MC prediction + ε-greedy improvement
SARSA: Truncated TD prediction + ε-greedy improvement
-->

| Method | Policy Evaluation | Policy Improvement |
|---|---|---|
| **Policy Iteration** | Iterative policy evaluation (full convergence) | Greedy |
| **Value Iteration** | Truncated iterative evaluation (1 sweep) | Greedy |
| **MC Control** | Truncated MC prediction (1 episode) | ε-greedy |
| **SARSA** | Truncated TD prediction (1 step) | ε-greedy |

---

### Python Code — SARSA Agent

```python
def sarsa(env, gamma=1.0, init_alpha=0.5, min_alpha=0.01,
          alpha_decay_ratio=0.5, init_epsilon=1.0, min_epsilon=0.1,
          epsilon_decay_ratio=0.9, n_episodes=3000):
    """
    SARSA — On-policy TD control method.
    TD prediction ဖြင့် policy evaluation,
    decaying ε-greedy ဖြင့် policy improvement.
    Step တိုင်းမှာ Q-function ကို update လုပ်တယ်။
    """
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    # ε-greedy action selection
    select_action = lambda state, Q, epsilon: \
        np.argmax(Q[state]) if np.random.random() > epsilon \
        else np.random.randint(len(Q[state]))

    # Alpha နဲ့ epsilon schedules
    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    for e in tqdm(range(n_episodes), leave=False):
        state, done = env.reset(), False
        # Initial state အတွက် action ကို ε-greedy ဖြင့် select
        action = select_action(state, Q, epsilons[e])

        while not done:
            # Environment ကို step
            next_state, reward, done, _ = env.step(action)

            # ⚡ SARSA ၏ key: next state အတွက် next action ကို select
            # (ε-greedy — actual next action ကို update တွင် သုံးမယ်)
            next_action = select_action(next_state, Q, epsilons[e])

            # TD target: reward + γ * Q(S', A')
            td_target = reward + gamma * Q[next_state][next_action] * (not done)
            td_error = td_target - Q[state][action]

            # Q-function update
            Q[state][action] = Q[state][action] + alphas[e] * td_error

            # next step အတွက် state, action update
            state, action = next_state, next_action

        # Episode ပြီးတိုင်း save
        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, Q_track, pi_track
```

> **SARSA** က **on-policy** method ဖြစ်တယ် — agent ရဲ့ behavior policy (ε-greedy) ကိုပဲ evaluate + improve လုပ်တယ်။ Update တိုင်းမှာ next action ကို actual behavior policy ကနေ select လုပ်တဲ့အတွက် "learning on the job" နဲ့ တူပါတယ်။

---

### RL အသုံးအနှုန်း — Batch vs. Offline vs. Online

| အသုံးအနှုန်း | အဓိပ္ပါယ် |
|---|---|
| **Batch learning** | Experience samples ကို fixed/advance ရှိပြီးသား data ကနေ synchronously သင်ယူခြင်း (fitting methods)။ Growing batch methods — data ကိုလည်း collect လုပ်ပြီး batch ကိုလည်း "grow" လုပ်ခြင်း။ |
| **Offline learning** | Interaction နှင့် learning ကို interleave လုပ်ပေမယ့်၊ performance ကို samples collect ပြီးမှ optimize လုပ်ခြင်း။ MC methods ဟာ episode-to-episode basis ဖြင့် learn/interact interleave ဖြစ်တဲ့အတွက် offline method ဟု ယူဆနိုင်တယ်။ |
| **Online learning** | Experience ကို ရရှိတာနဲ့ ချက်ချင်း step တိုင်းမှာ သင်ယူခြင်း။ TD methods ဟာ online method ဖြစ်တယ်။ |

---

## ၆.၇ — Behavior နှင့် Learning ကို Decouple လုပ်ခြင်း (Off-policy Learning)

SARSA ၏ TD update equation ကို ပြန်ကြည့်ကြမယ်:

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \Big[ R_{t+1} + \gamma Q(S_{t+1}, \underbrace{A_{t+1}}_{\text{actual next action}}) - Q(S_t, A_t) \Big]$$

$A_{t+1}$ နေရာမှာ **ဘယ် action ကို ထည့်လဲ** ဆိုတာ ပိုစဉ်းစားကြည့်ပါ — ဒါက RL ရဲ့ အရေးအကြီးဆုံး advancement တစ်ခုဆီ ခေါ်သွားပါလိမ့်မယ်...

---

## ၆.၈ — Q-learning: Optimal ဖြစ်အောင် သင်ယူခြင်း (ရွေးချယ်မှု မတူရင်တောင်)

### On-policy vs. Off-policy

**SARSA** ဟာ "learning on the job" ပါ — agent က ကိုယ့် behavior policy ကိုပဲ evaluate + improve လုပ်တယ် (on-policy)။ ကိုယ့်ရဲ့ current mistakes ကနေသာ သင်ယူတယ်။

**Q-learning** ကတော့ "learning from others' experiences" ပါ — agent က behavior policy (data generate လုပ်တဲ့ policy) မဟုတ်ဘဲ **target policy** (optimal greedy policy) ကို learn လုပ်တယ် (off-policy)။ Agent က randomly explore လုပ်နေလည်း optimal policy ကို approximate လုပ်နိုင်တယ်။

### Q-learning Update Equation

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \Big[ R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t) \Big]$$

### SARSA vs. Q-learning — Key Difference

| | SARSA | Q-learning |
|---|---|---|
| **Target ထဲက action** | $Q(S_{t+1}, A_{t+1})$ — actually taken action | $\max_{a'} Q(S_{t+1}, a')$ — max over all actions |
| **Policy type** | On-policy | Off-policy |
| **Learning about** | Behavior policy (ε-greedy) | Optimal policy (greedy) |

**SARSA:** "ငါ ဘာလုပ်နေလဲ — ဒီ action ယူတဲ့ policy ဘယ်လောက်ကောင်းလဲ"

**Q-learning:** "ငါ ဘာပဲလုပ်နေနေ — **အကောင်းဆုံး** action ယူရင် ဘယ်လောက်ရနိုင်လဲ"

---

### Python Code — Q-learning Agent

```python
def q_learning(env, gamma=1.0, init_alpha=0.5, min_alpha=0.01,
               alpha_decay_ratio=0.5, init_epsilon=1.0, min_epsilon=0.1,
               epsilon_decay_ratio=0.9, n_episodes=3000):
    """
    Q-learning — Off-policy TD control method.
    Behavior policy (ε-greedy) နဲ့ data collect ပေမယ့်
    target policy (greedy/optimal) ကို learn လုပ်တယ်။
    """
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state, Q, epsilon: \
        np.argmax(Q[state]) if np.random.random() > epsilon \
        else np.random.randint(len(Q[state]))

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    for e in tqdm(range(n_episodes), leave=False):
        state, done = env.reset(), False

        while not done:
            # ε-greedy ဖြင့် action select (behavior policy)
            action = select_action(state, Q, epsilons[e])
            next_state, reward, done, _ = env.step(action)

            # ⚡ Q-learning ၏ key: max over all actions in next state
            # (target policy = greedy — actual next action ကို မသုံး!)
            td_target = reward + gamma * Q[next_state].max() * (not done)
            td_error = td_target - Q[state][action]
            Q[state][action] = Q[state][action] + alphas[e] * td_error

            state = next_state  # next_action ကို track မလုပ်ရ

        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))

    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, Q_track, pi_track
```

> **Q-learning** ဟာ model-free, off-policy, bootstrapping method — agent randomly act နေလည်း optimal Q-function ကို approximate လုပ်နိုင်ပါတယ်!

---

### Miguel ၏ Analogy — လူသားတွေလည်း On-policy/Off-policy သင်ယူတယ်

> Miguel ၏ သားက **on-policy learner** — ကစားစရာနဲ့ ကိုယ်တိုင်ရုန်းပြီး သင်ယူချင်တယ်၊ ဖေဖေလာပြရင် မကျေနပ်ဘူး။ ကိုယ့်အတွေ့အကြုံကနေပဲ တတ် မြောက်ချင်တယ်။
>
> သမီးကတော့ **off-policy learner** — ဖေဖေ လုပ်ပြတာကို ကြည့်ပြီးမှ ကိုယ်တိုင် စမ်းကြည့်တယ်။

On-policy vs. off-policy နှစ်ခုလုံး pros/cons ရှိပါတယ်:
- **On-policy** — intuitive, stable (ပီယာနိုပဲ တတ်ချင်ရင် ပီယာနိုပဲ လေ့ကျင့်)
- **Off-policy** — multiple sources ကနေ သင်ယူနိုင် (meditation က ပီယာနိုတီးရာမှာ ကူညီကောင်း ကူညီနိုင်တယ်!)

> ⚠️ **သတိထားရန်:** Off-policy learning + bootstrapping + function approximation — ဒီ ၃ ခု ပေါင်းရင် **divergence** ဖြစ်နိုင်ကြောင်း သက်သေပြပြီးသားပါ။

---

### RL သဘာဝစကား — On-policy vs. Off-policy Learning

| | On-policy | Off-policy |
|---|---|---|
| **အဓိပ္ပါယ်** | Decision making အတွက် သုံးနေတဲ့ policy ကိုပဲ evaluate/improve လုပ်ခြင်း | Data generate လုပ်တဲ့ policy နဲ့ learn about လုပ်တဲ့ policy **မတူ**ခြင်း |
| **Policy အရေအတွက်** | Policy ၁ ခု (behavior = target) | Policy ၂ ခု (behavior policy μ ≠ target policy π) |
| **ဥပမာ** | MC Control, SARSA | Q-learning, Double Q-learning |

---

## ၆.၉ — Convergence Theories

### GLIE — Greedy in the Limit with Infinite Exploration

**On-policy** algorithms (MC Control, SARSA) အတွက် optimal policy ဆီ converge ဖြစ်ဖို့ သတ်မှတ်ချက် ၂ ခု ရှိတယ်:

1. **State-action pairs အားလုံးကို infinitely often explore လုပ်ရမည်**
2. **Policy သည် greedy policy ဆီ converge ဖြစ်ရမည်** (ε → 0)

> ε-greedy ကို ကျသွားစေရင် — ε ကို အရမ်းမြန်ကျစေရင် condition 1 မဖြည့်စွမ်းနိုင်၊ အရမ်းနှေးကျစေရင် converge အချိန်ကြာမယ်။

**Off-policy** algorithms (Q-learning) အတွက် — condition 1 သာ လိုအပ်တယ်! Condition 2 မလိုတော့ — target policy က behavior policy နဲ့ မတူတဲ့အတွက်ဖြစ်ပါတယ်။

### Stochastic Approximation Theory — Learning Rate Requirements

Sample-based learning မှာ variance ရှိတဲ့အတွက် converge ဖြစ်ဖို့ learning rate ကို ၀ ဆီ push လုပ်ရမည်:

$$\sum_{t=1}^{\infty} \alpha_t = \infty \quad \text{(learning rate ဘယ်တော့မှ ၀ မဖြစ်ရ)}$$

$$\sum_{t=1}^{\infty} \alpha_t^2 < \infty \quad \text{(noise ကိုလည်း ထိန်းချုပ်နိုင်ရမယ်)}$$

ဥပမာ — $\alpha_t = 1/t$ ဆိုရင် အစမှာ large (single sample ကို tight follow မလုပ်ရအောင်)、နောက်မှ small (signal ကို noise ကနေ ခွဲထုတ်ရအောင်)。

> **Practice မှာတော့** — small constant learning rate ကို common ဖြင့် သုံးတယ်။ Non-stationary environments မှာ constant αက ပိုကောင်းတယ်။

---

## ၆.၁၀ — Double Q-learning: Maximization Bias ကို ဖြေရှင်းခြင်း

### Maximization Bias ပြဿနာ

Q-learning ဟာ value function ကို **overestimate** လုပ်တတ်ပါတယ်။ Step တိုင်းမှာ next state ရဲ့ action-value estimates ထဲက **maximum** ကို ယူတယ်。 ပေမယ့် ကျွန်တော်တို့ လိုအပ်တာက actual maximum value ပါ။ ကျွန်တော်တို့ လုပ်နေတာက:

$$\text{estimates ရဲ့ maximum} \neq \text{maximum ရဲ့ estimate}$$

Estimates တွေမှာ bias ရှိတယ် — positive bias ရောနိုင်、negative bias ရောနိုင်. Max ယူခြင်းက positive bias ကိုပဲ favor လုပ်ပြီး errors compound ဖြစ်စေတယ်. ဒါကို **maximization bias** လို့ ခေါ်ပါတယ်။

> **ဥပမာ:** Actual values အကုန် 0 ဖြစ်ပေမယ့် estimates က `[0.11, 0.65, -0.44, -0.26, ...]` ဆိုရင်、actual max = 0 ပေမယ့် max(estimates) = 0.65 ။ Max ယူရင် positive bias ပါတဲ့ value ကိုပဲ အမြဲရွေးမိပြီး compound error ဖြစ်ပါတယ်。

### Double Learning Solution

**Q1** နဲ့ **Q2** — $Q$-function ၂ ခု ကို track လုပ်ပါတယ်:

- Step တိုင်းမှာ coin flip — Q1 update ရမလား Q2 update ရမလား ဆုံးဖြတ်
- **Q1 update ရင်:** Q1 ကနေ best action ကို ရွေး ($a^* = \arg\max_a Q_1(S_{t+1}, a)$)၊ ပေမယ့် **Q2** ကနေ ဒီ action ရဲ့ value ကို ယူ
- **Q2 update ရင်:** Q2 ကနေ best action ကို ရွေး ($a^* = \arg\max_a Q_2(S_{t+1}, a)$)၊ ပေမယ့် **Q1** ကနေ ဒီ action ရဲ့ value ကို ယူ

### Double Q-learning Update Equations

**Q1 ကို update ရင်:**

$$a^* = \arg\max_a Q_1(S_{t+1}, a)$$

$$Q_1(S_t, A_t) \leftarrow Q_1(S_t, A_t) + \alpha \Big[ R_{t+1} + \gamma \, Q_2(S_{t+1}, a^*) - Q_1(S_t, A_t) \Big]$$

**Q2 ကို update ရင် (mirror):**

$$a^* = \arg\max_a Q_2(S_{t+1}, a)$$

$$Q_2(S_t, A_t) \leftarrow Q_2(S_t, A_t) + \alpha \Big[ R_{t+1} + \gamma \, Q_1(S_{t+1}, a^*) - Q_2(S_t, A_t) \Big]$$

**Action selection:** Q1 + Q2 ၏ mean ကို သုံး:

$$\pi(s) = \arg\max_a \frac{Q_1(s,a) + Q_2(s,a)}{2}$$

> Cross-validation pattern နဲ့ တူပါတယ် — Q-function တစ်ခုက ရွေးတဲ့ action ကို တစ်ခြား Q-function ဖြင့် validate လုပ်ခြင်းဖြစ်ပါတယ်。 ဒါကြောင့် positive bias ကို ထိန်းချုပ်နိုင်ပါတယ်。

---

### Python Code — Double Q-learning Agent

```python
def double_q_learning(env, gamma=1.0, init_alpha=0.5, min_alpha=0.01,
                      alpha_decay_ratio=0.5, init_epsilon=1.0, min_epsilon=0.1,
                      epsilon_decay_ratio=0.9, n_episodes=3000):
    """
    Double Q-learning — Q-function ၂ ခု ဖြင့် maximization bias ကို mitigate.
    Coin flip ဖြင့် Q1/Q2 update ကို alternate, action select မှာ mean ကို သုံး။
    """
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []

    # Q-function ၂ ခု initialize
    Q1 = np.zeros((nS, nA), dtype=np.float64)
    Q2 = np.zeros((nS, nA), dtype=np.float64)
    Q_track1 = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    Q_track2 = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action = lambda state, Q, epsilon: \
        np.argmax(Q[state]) if np.random.random() > epsilon \
        else np.random.randint(len(Q[state]))

    alphas = decay_schedule(init_alpha, min_alpha, alpha_decay_ratio, n_episodes)
    epsilons = decay_schedule(init_epsilon, min_epsilon, epsilon_decay_ratio, n_episodes)

    for e in tqdm(range(n_episodes), leave=False):
        state, done = env.reset(), False

        while not done:
            # ⚡ Action select: Q1 + Q2 ရဲ့ mean ကို သုံး
            action = select_action(state, (Q1 + Q2) / 2.0, epsilons[e])
            next_state, reward, done, _ = env.step(action)

            # Coin flip: Q1 update ရမလား Q2 update ရမလား
            if np.random.randint(2):
                # ---- Q1 update ----
                # Q1 ကနေ best action ရွေး
                argmax_Q1 = np.argmax(Q1[next_state])
                # Q2 ကနေ value ယူ (cross-validation!)
                td_target = reward + gamma * Q2[next_state][argmax_Q1] * (not done)
                td_error = td_target - Q1[state][action]
                Q1[state][action] = Q1[state][action] + alphas[e] * td_error
            else:
                # ---- Q2 update (mirror) ----
                argmax_Q2 = np.argmax(Q2[next_state])
                td_target = reward + gamma * Q1[next_state][argmax_Q2] * (not done)
                td_error = td_target - Q2[state][action]
                Q2[state][action] = Q2[state][action] + alphas[e] * td_error

            state = next_state

        Q_track1[e] = Q1
        Q_track2[e] = Q2
        pi_track.append(np.argmax((Q1 + Q2) / 2.0, axis=1))

    # Final Q, V, policy — mean of Q1, Q2
    Q = (Q1 + Q2) / 2.0
    V = np.max(Q, axis=1)
    pi = lambda s: {s:a for s, a in enumerate(np.argmax(Q, axis=1))}[s]

    return Q, V, pi, (Q_track1 + Q_track2) / 2.0, pi_track
```

> **အသိ:** Experience ကို Q-function ၂ ခု ကြား ခွဲပေးရတဲ့အတွက် training အနည်းငယ် နှေးနိုင်ပါတယ်。

---

## ၆.၁၁ — SWS Environment တွင် စမ်းသပ်ခြင်း (Experimental Results)

Algorithms ၄ ခုစလုံးကို **hyperparameters တူတူ** ဖြင့် SWS environment မှာ run ပါတယ်:
- $\gamma$ (discount), $\alpha$ (learning rate), $\epsilon$ (exploration) နှင့် respective decaying schedules တူတူ
- Episodes — 3,000
- Alpha → 0.01 ဆီ decay (fully converge ဖြစ်ဖို့)
- Epsilon → 0.1 ဆီ decay (practice မှာ 0 ဆီ decay ခဲသေးတယ်)

<!-- 📖 စာအုပ်ပုံ - Value function estimates comparison (p.197)
FVMC, SARSA, Q-learning, Double Q-learning value estimates over episodes
-->

### Value Function Estimates

| Algorithm | Observation |
|---|---|
| **FVMC** | Estimates တွေမှာ **high variance** — ခုန်ပြီး ကျတယ် (MC prediction ကဲ့သို့) |
| **SARSA** | MC ထက် **variance နည်း** ပေမယ့် optimal values ဆီ ရောက်ဖို့ MC နဲ့ အချိန်တူတူလိုချင်တယ် |
| **Q-learning** | Optimal values ဆီ **အမြန်ဆုံး** track ပေမယ့်、estimates တွေ **overestimate** ဖြစ်ပြီး aggressively jump around |
| **Double Q-learning** | Q-learning ထက် **အနည်းငယ်နှေး** ပေမယ့် **ပိုပို stable** — overestimation ရှိသေးပေမယ့် controlled |

<!-- 📖 စာအုပ်ပုံ - Policy success rate, mean return, regret comparison (p.198) -->

### Policy Performance

| Metric | FVMC | SARSA | Q-learning | Double Q-learning |
|---|---|---|---|---|
| **100% success rate** | နှေး | နှေး | Q-learning ရရှိသော်လည်း overshoot | **အမြန်ဆုံး** |
| **Mean return** | Optimal ကနေ ဝေး | Optimal ကနေ ဝေး | ကောင်း ပေမယ့် noisy | **Optimal ကို အမြန်ဆုံး track** |
| **Regret** (reward left on table) | မြင့် | မြင့် | အလယ်အလတ် | **အနိမ့်ဆုံး** |

<!-- 📖 စာအုပ်ပုံ - Value estimation errors comparison (p.199) -->

### Value Function Estimation Error

| Metric | Observation |
|---|---|
| **V-function MAE** | Q-learning က near-zero ဆီ အမြန်ဆုံး drop ပေမယ့်、Double Q-learning က lowest error ကို first ရောက် |
| **Q-function MAE** | SARSA နှင့် FVMC comparably slow、Q-learning fast but overshoots、Double Q-learning best overall |

> **Key Finding:** Double Q-learning ဟာ Q-learning ထက် stable ဖြစ်ပြီး optimal policy ကို faster ရောက်ပါတယ်။

---

## ၆.၁၂ — Key Equations Summary

| Equation | Formula |
|---|---|
| **MC Return** | $G_t = \sum_{k=0}^{T-t-1} \gamma^k R_{t+k+1}$ |
| **MC Control Update** | $Q(s,a) \leftarrow Q(s,a) + \alpha [G_t - Q(s,a)]$ |
| **SARSA Update** | $Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha [R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)]$ |
| **Q-learning Update** | $Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha [R_{t+1} + \gamma \max_{a'} Q(S_{t+1},a') - Q(S_t,A_t)]$ |
| **Double Q-learning (Q1)** | $Q_1(S_t,A_t) \leftarrow Q_1 + \alpha [R_{t+1} + \gamma \, Q_2(S_{t+1}, \arg\max_a Q_1(S_{t+1},a)) - Q_1(S_t,A_t)]$ |
| **GLIE condition** | $\epsilon \to 0$ as $t \to \infty$, all $(s,a)$ explored infinitely |
| **α convergence** | $\sum \alpha_t = \infty, \quad \sum \alpha_t^2 < \infty$ |

---

## ၆.၁၃ — Algorithms Comparison

| Feature | MC Control | SARSA | Q-learning | Double Q-learning |
|---|---|---|---|---|
| **Policy Evaluation** | MC (full episode) | TD (one-step) | TD (one-step) | TD (one-step) |
| **Update Timing** | Episode ပြီးမှ | Step တိုင်း | Step တိုင်း | Step တိုင်း |
| **On/Off-policy** | On-policy | On-policy | Off-policy | Off-policy |
| **Bootstrapping** | No | Yes | Yes | Yes |
| **Overestimation** | Low | Low | **High** | **Mitigated** |
| **Variance** | **High** | Lower | Moderate | Low |
| **Convergence** | Moderate | Moderate | Fast (but overshoots) | Best overall |

---

## ၆.၁၄ — နိဂုံးချုပ် (Summary)

ဒီအခန်းမှာ ယခင်သင်ယူခဲ့တဲ့ အရာအားလုံးကို practice ထဲ ထည့်ခဲ့ပါတယ်။ Trial-and-error learning ဖြင့် policy optimize လုပ်တဲ့ algorithm တွေကို သင်ယူခဲ့ပါတယ်။ Sequential ရော evaluative ရော ဖြစ်နေတဲ့ feedback ကနေ သင်ယူရတဲ့ agents — immediate/long-term goals ကို balance လုပ်ရင်း information ကိုလည်း gather/utilize ရင့် ဖြေရှင်းရပါတယ်။

**Key takeaways:**

1. **GPI pattern** — Policy evaluation + improvement interaction ဖြင့် optimal policy ဆီ iteratively converge
2. **MC Control** — Episode ပြီးမှ update, high variance but unbiased
3. **SARSA** — On-policy TD, step-by-step update, stable
4. **Q-learning** — Off-policy TD, optimal policy ကို learns regardless of behavior policy
5. **Double Q-learning** — Maximization bias ကို mitigate, more stable convergence
6. **GLIE + Stochastic Approximation** — Convergence theory requirements
7. **On-policy vs. Off-policy** — Pros/cons; off-policy + bootstrapping + function approximation → possible divergence

**ယခုအခန်းပြီးနောက် သင်သိပြီး ဖြစ်ရမည့် အချက်များ:**

- RL agents အများစု **GPI** pattern ကို follow လုပ်ကြတယ်
- GPI က **policy evaluation + improvement** ဖြင့် control problem ကို solve လုပ်တယ်
- GPI pattern ကို follow လုပ်တဲ့ agents အမျိုးမျိုးကို သင်ယူခဲ့ပြီ
- MC control, SARSA, Q-learning, double Q-learning — algorithm တစ်ခုချင်းစီ၏ ကွာခြားချက်များကို နားလည်ပြီ
- On-policy vs. off-policy, online vs. offline ဆိုတဲ့ concept တွေကို နားလည်ပြီ
- Convergence theories (GLIE, stochastic approximation) ကို သိရှိပြီ

**နောက်အခန်းမှာ** — ပိုမိုရှုပ်ထွေးတဲ့ environments မှာ ပိုထိရောက်ပြီး ပိုမိုစွမ်းဆောင်ရည်ကောင်းတဲ့ control methods တွေကို လေ့လာမယ်။ ယခု chapter ရဲ့ methods ထက် **experience samples နည်းနည်း** ဖြင့် environments တွေကို solve လုပ်နိုင်တဲ့ advanced methods ဖြစ်ပါတယ်။

---

### Tweetable Feats — ကိုယ်တိုင် စမ်းသပ်ကြည့်ရန်

1. **#gdrl_ch06_tf01:** Learning rate ($\alpha$) နဲ့ discount factor ($\gamma$) — ဒီ variables ၂ ခု ဘယ်လို interact လုပ်လဲ? Total reward နဲ့ policy success rate ကို ဘယ်လို affect လုပ်လဲ?

2. **#gdrl_ch06_tf02:** ε-greedy ပဲ မဟုတ်ဘဲ Chapter 4 က exploration strategies တွေကို ဘယ်လို သုံးမလဲ? ကိုယ်ပိုင် exploration strategy ဖန်တီးပြီး test လုပ်ကြည့်ပါ!

3. **#gdrl_ch06_tf03:** ဒီအခန်းရဲ့ algorithms တွေက time step limit ကို မှန်ကန်စွာ မသုံးထားပါ — ဘာကို ရည်ညွှန်းနေလဲ ရှာဖွေပြီး ပြင်ဆင်ကြည့်ပါ!

4. **#gdrl_ch06_tf04:** ဒီအခန်းနဲ့ ပတ်သက်ပြီး ကိုယ်တိုင် investigate လုပ်ချင်တာ ဘာပဲဖြစ်ဖြစ် share ပါ!

---
