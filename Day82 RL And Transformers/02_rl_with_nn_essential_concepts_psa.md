### psa summarized
#### Neural Network and Reinforcement Learning
- NN မှာ ground truth output မရှိတဲ့ အခါမှာလည်း NN ကို rl နဲ့ train လို့ရတယ်။
- အဲ့ထဲကမှ Policy Gradient ကိုသုံးမယ်။

#### Policy Gradient
- ပုံမှန် NN မှဆိုရင် derivative က - ဖြစ်ရင် weight ကို တိုးရမယ်။ + ဖြစ်နေရင်တော့ ဘယ်ဘက်ကို တိုးရမယ်။ ( BP ရဲ့ Gradient descent ) ပေါ့။ ခုကျတော့ output မရှိတော့ gradient မရှိဘူးလေ။
- အဲ့တော့ တခုခုကို ခန့်မှန်းပြီးမှ အဲ့ကောင်ကို derivative ရှာမယ်လေ။ 
- ဘယ်လိုလဲဆိုတော့ FF တွက်လိုက်လို့ P(A) = 0.5, 1 - P(A) = 0.5 ရတယ်ဆိုပါစို့။ Number line မှာဆို တဝက်စီပေါ့။ အဲ့မှာ random တစ်ခုကို ထုတ်လိုက်ပြီး P(A) ဘက်ရဲ့ 0 နဲ့ 0.5 ကြားမှာရတယ်။ 0.2 ရတယ်ဆိုပါစို့။ P(A) ဆိုင်ကိုပဲ သွားရမှာဖြစ်ပါတယ်။  အဲ့ဆိုင်ကို သွားတာက တကယ် မှန်တဲ့ case ဆိုရင် P(A) = 1 ဖြစ်ဖို့လိုတာပေါ့။ 1 - P(A) = 0 ဖြစ်ဖို့လိုတာပေါ့။ ( case မှန်ရင် reward = 1 ( may be any positive ) ပေးပြီး မှားရင် reward = -1 ( may be any negative ) ပေးပါမယ်။)
- ဒါဆို Idea value အဖြစ် P(A) = 1, Actual value အဖြစ် P(A) = 0.5 ( network's output ) ဖြစ်ပါတယ်။ 
- အဲ့ဒီ ခြားနားခြင်းကို  respect to Bias အရ derivative တွက်ပါမယ်။ 
- အဲ့ဒီ တွက်ထားတဲ့ derivative ကို reward နဲ့ မြှောက်ပြီး update လုပ်ပါမယ်။ ( derivative * reward = updated derivative ) plus/minux
- အရေးကြီးတာက negative reward( -1 ), incorrect guess ( 0.5 ) က လာတဲ့ derivative ကို -1 * 0.5 = -0.5 ဖြစ်တဲ့ အတွက် flip လုပ်ပြီးမှ correction လုပ်ရမှာဖြစ်ပါတယ်။ 
- ဆိုတော့ updated derivative သည် +ve slope or -ve slope ပြောနေတာမို့ bias ကို ရှေ့တိုးမလား/နောက်ဆုတ်မလား မြင်သာစေပါတယ်။ 
- သူ့ထပ် ပိုမှန်တဲ့ အနေအထားရှိမယ်ဆိုရင် reward = 2 လည်းဖြစ်နိုင်တယ်ဆိုပါစို့။  derivative direction မပြောင်းဘဲ slope ပဲတိုးလာမယ်။ step များလာမယ်လို့ယူဆရမှာပဲ။
- အဲ့လိုပဲ reward = -2 အတွက်လည်း ရှိနိုင်တာမို့ သူလည်း သဘောတရားအတူတူပါပဲ။

##### Gradient တွက်မယ်။
- အဲ့ဒီ updated derivative ကို lr နဲ့ မြှောက်ပြီး step size တွက်မယ်။ ( step_size = learning_rate * updated_derivative )
- အဲ့တာနဲ့ Bias ကို update လုပ်မယ်။ ( Bias = old_bias - step_size )
- bias update ပြီးတော့ FF ပြန်တွက်တာပါပဲ။ ဒီခါတွက်ရင်တော့ P(A) = 0.6 ရပြီး တိုးလာမှာပေါ့။ 

