### psa summarized
- #### MDP ( Markov Decision Process )
    - python rl framework
    - model တွေက decision ချနိုင်ဖို့
        - partially random
        - partially control by agen
    - discrete time မှာ အလုပ်လုပ်, state, action , new_state, reward
    - markov properties --> future သည် ့history ကို မမှီခိုပဲနဲ့ current state နဲ့ သူ့ကြောင့်ဖြစ်တဲ့ action ကို ပဲ မှီခိုတယ်။
    - uncetrain environment မှာ plan ချဖို့ markov properties ရဲ့ အကူအညီနဲ့သွားမယ်။
    - finite MDP ( limited လုပ်ထားတဲ့ state နဲ့ action )
    - MDP ရဲ့ အရေးကြီး Components ၄ ခု 
        - 1.  S = state
        - 2.  A(S) = action , set of Actions
        - 3.  T = transition function model ( s, a , s' )
        s' ဆိုတာက လက်ရှိ state s ကို action တစ်ခုလုပ်လိုက်လို့ နောက်ထပ်ရလာတဲ့ probability of state ပါ။ 
        T ( s, a ,s' ) = P ( s' | s, a )
        - 4.  Reward function, 
                - state ကြောင့်ဖြစ်လာတဲ့ Reward = R(s), 
                - state နဲ့ action ကြောင့်ဖြစ်တဲ့ Reward = R(s,a), 
                - နောက်တော့ state, action, ရလာတဲ့ state ပါဝင်တဲ့ Reward = R(s,a,s')
    - positive reward, negative reward ပေးပြီး cumulative ဖြစ်လာမယ်။

- #### MDP ( S, T, A, R )
    - သူက environment တစ်ခုမှာ decision making လုပ်တဲ့ နည်းလမ်းတစ်ခုပဲ
    - သူ့မှာ Sequence ( အကျိုးဆက် ဖြစ်မယ်။ )
    - Stochastic ( Probability ရှိမယ်။ )
    - Goal Driven ဖြစ်မယ်။ ( reward တွေကြောင့် )
    - နောက် အပိုင်း ၄ ခုမှာ ပထမဆုံးကတော့ State , s ပေါ့။ ဥပမာ 4x5 grid world မှာ ဆို state 12 ခုရှိတယ်။ 
    - နောက် Action, A(s)
    - တတိယတခုက Transition , သူ့ကို Model လို့လဲခေါ်တယ်။ Transition Model ပေါ့။ T (s,a,s') = P(s'|a,s)
    - Reward Function , R(s), R(s,a), R(s,a,s')

- #### Policy
    - State ကနေပြီးတော့ action တစ်ခုကို သုံးရမယ်ဆိုရင် agent ကို ဘာလုပ်မလဲဆိုပြီးခိုင်းတာနဲ့ အတူတူပါပဲ။
    - တနည်းအားဖြင့် state ကနေ အကောင်းဆုံး Action တစ်ခုကို mapping လုပ်ပေးတာပဲ။ ( Structure )
    - MTP Policy ရဲ့ Goal သည် final state ကို ရောက်ရုံတင်မဟုတ်ပဲ Maximum Reward Long term ရဖို့ဖြစ်တယ်။ ( Goal )
    - နောက်သူ့ရဲ့ Environment သည် Stochastic ဖြစ်ပြီး probability တွေပဲဖြစ်တယ်။ ( Environment )
    - သူ့ရဲ့ Behavior ကတော့ state by state ( Auto ) သွားတာဖြစ်ပါတယ်။ ( Behavior )

- #### MDP Policy Breakdown
    - Uncertainty in outcomes, actions တွေက Probability တွေလေ၊ ဒါကြောင့် outcome မှာ မသေချာမှုရှိတာပေါ့။ 
    - sequence of actions မှာလဲ အဲ့လို မသေချာမှုရှိနိုင်တယ်။
    - ဒီ model သည် One Step Decision Model ဖြစ်ပါတယ်။ ( current state ပေါ်မူတည်ပြီး action ကို တွက်တာ )
    - Memory Less ( Markov Properties ) history အားလုံးကို လိုက်ကြည့်မနေဘဲ current state ကို ရောက်နေချိန်မှာ best actions ကို ဒီမှာပဲလုပ်မှာ။ လက်ရှိဘာလုပ်ရမလဲဆိုတာပဲ အာရုံစိုက်။ stochastic outcome မို့လို fixed plan လိုမဟုတ်ဘဲ policy နဲ့ပဲထိန်းတယ်။ မှားခဲ့ရင်လည်း current ရောက်နေတဲ့ နေရာကပဲဆက်တွက်ပြီး goal ကို ဆက်သွားတယ်။

- #### MDP Rewards
    - Immediate Rewards vs Delayed Rewards
    - Agent က Action ကို ချက်ချင်း change လုပ်လိုက်တာကြောင့် Immediate rewards ရသွားတယ်။ s' ရောက်သွားတယ်ပေါ့။ တကယ့် real world မှာ agent ရဲ့ လက်ရှိ လုပ်ဆောင်ချက်ဟာ မှန်နေတယ်/မှားနေတယ် ပြောလို့မရဘူး။ နောက်ပိုင်းမှာ ကြည့်ပြီး သိသာနိုင်တာပါ။
    - Agent သည် Delayed feedback ကနေ learning လုပ်ရမှာဖြစ်ပါတယ်၊
    - Temporal Credit Assignment ( value function, policy gradient, exp replay )
