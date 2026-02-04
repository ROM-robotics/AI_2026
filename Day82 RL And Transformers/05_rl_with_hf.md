### psa summarized

#### align pre-trained model
- supervised fine tuning
- rl with human feedback 

#### supervised fine tuning
- datasets with human created prompt and user created responses
- အဲ့လို train ရင် supervised ft လုပ်ထားတဲ့ model ကို ရမယ်။
- generalize model မဖြစ်ပဲ over fitted model ဖြစ်နိုင်တယ်။ 
- အဲ့လို မဖြစ်ဖို့ အရမ်းကြီးတဲ့ dataset လိုတယ်။ ဒါက မဖြစ်နိုင်လို့ rl နဲ့ သုံးမယ်။ 

#### rl with human feedback
- user က model ကို မေးလိုက်ရင် model က response pair အတွဲတွေဖန်တီးပြီး အဲ့ဒီ response တွေထဲက ဘယ်အဖြေက ပိုကောင်းသလဲဆိုတာ လူတွေကို ရွေးချယ်ခိုင်းမယ်။ လူတွေကို response ကို မရေးခိုင်းဘဲ ရွေးပဲရွေးခိုင်းလိုက်တဲ့သဘောပါ။ ဒါကိုပဲ rl with human feedback လို့ခေါ်တာပေါ့။
- အဲ့ဒီ လူတွေပေးလိုက်တဲ့ preference data နဲ့ high score ရအောင် train တာပါပဲ။

#### train the reward ( model with supervised ft for calculating reward )
- ဘယ်လိုလုပ်လဲဆိုတော့ supervised fine tuning လုပ်ထားတဲ့ ငါတို့ရဲ့ model ကို copy တစ်ခုကူးမယ်။
- အဲ့ဒီ decoder model ရဲ့ Q,V,K attention block ပြီးရင် အဖြေထုတ်ပေးတဲ့ unembedding Layer ကို ဖြုတ်လိုက်မယ်။ 
- အဲ့ဒီနေရာမှာ reward ထုတ်ဖို့ single output ကို ပဲ အစားထိုးပါမယ်။
- သဘောက attention block ကထွက်လာတဲ့ အဖြေကို reward ထုတ်ဖို့ သုံးတယ်ပေါ့။ ဒါဆို အခု model သည် reward ထုတ်ပေးတဲ့ model ဖြစ်သွားပါပြီ။ 
- နောက်တော့ Prompt ( Question ) နဲ့ response ( bad answer, good answer ) တို့ကို ပေးပြီး reward model ကို reward + , reward - ရအောင် train ပါတယ်။ 
- 2022 မှာ open AI က သုံးတဲ့ hf နဲ့ train တဲ့ manuscript ထဲက loss function က ဒီလိုပါ။
- -1 * log( Sigmoid( reward_better - reward_worse ) )
- difference term ကို ကြည့်ရင် reward_better ဆိုတာက prompt, good response နဲ့ reward + ရအောင် train ထားတဲ့ကောင်ပါ။
- ဆိုတော့ ( reward_better - reward_worse ) က huge positive number ပါ။ 
- ဒါကို sigmoid ယူတော့ 0 နဲ့ 1 ကြားရမှာပေါ့။
- ဒါကို log() ယူတော့ high value ပြန်ရပါမယ်။
- အဲ့ log() ကို gradient descent က lowest curve position ရှာမှာဖြစ်တယ်။
- အဲ့ဒီ lowest point က ငါတို့လိုချင်တာနဲ့ ပြောင်းပြန်ဖြစ်တာမို့ -1 နဲ့ မြှောက်လိုက်တာဖြစ်တယ်။ curve ကို flip လုပ်လိုက်တာလို့ ပြောလို့ရပါတယ်။ 

#### training original model with reward model
- အဲ့အချိန်မှာ supervised ft dataset ထဲကမဟုတ်တဲ့ data သုံးမယ်။ human provided response လဲ မသုံးဘူး။ prompt သက်သက်ပဲ ပေးမယ်ဆိုရင် original model က response ထုတ်ပေးမှာဖြစ်ပါတယ်။ 
- အဲ့လို response ထွက်လာတဲ့အခါ prompt, response ကို reward model ကို ပေးရင် reward model က reward တွက်ထုတ်မှာဖြစ်တယ်။ ဥပမာ supervised model ရဲ့  response က မကောင်းတဲ့ အဖြေဆိုရင် reward = -2 တွက်ထုတ်လိုက်တယ်ဆိုပါစို့။
- အဲ့ဒီ reward နဲ့ original model ကို ထပ်ပြီး train ရမှာဖြစ်ပါတယ်။
- အဲ့လို train ပြီး reward သည် high positive အဖြေရအောင် ပြုလုပ်ရမှာဖြစ်ပါတယ်။
- ဒါမှသာ original model သည် polite and helpful response များကို ထုတ်လုပ်ပေးနိုင်မှာဖြစ်ပါတယ်။ 
- ဒါဆိုရင် super huge ဖြစ်တဲ့ supervised fine tuning dataset ကို သုံးစရာမလိုတော့ပဲ overfit ဖြစ်ခြင်းကို ရှောင်နိုင်ပြီဖြစ်ပါတယ်။ 
- အခုလို reward model နဲ့ train ပြီးတဲ့အခါ ကျတော်တို့ရဲ့ original model ကို လူတွေအသုံးပြုလို့ရတဲ့ model အဖြစ် အသုံးပြုနိုင်ပြီဖြစ်ပါတယ်။ 

### theory အရ
#### rl ကို NN နဲ့ သုံးတယ်ဆိုသော်လည်း တကယ်လက်တွေ့မှာတော့ Popular ဖြစ်တဲ့ 
#### PPO ( Proximal Policy Optimization )
#### GRPO ( Group Relative Policy Optimization) တို့နဲ့ သုံးကြပါတယ်။

#### ၁။ PPO (Proximal Policy Optimization)
PPO ကတော့ လက်ရှိ AI လောက (အထူးသဖြင့် ChatGPT လို LLM တွေ) မှာ အသုံးအများဆုံး RL နည်းလမ်းပါ။

- ဘယ်လိုအလုပ်လုပ်လဲ: Model က ထုတ်ပေးတဲ့ အဖြေတွေကို Reward Model တစ်ခုက အမှတ်ပေးပါတယ်။ အမှတ်ကောင်းရင် အဲဒီလမ်းကြောင်းကို ဆက်သွားဖို့ Model ကို သင်ပေးပါတယ်။

- ထူးခြားချက်: "Proximal" ဆိုတဲ့အတိုင်း Model ကို တစ်ခါတည်းနဲ့ အကြီးကြီး ပြောင်းလဲပစ်တာမျိုး မလုပ်အောင် ထိန်းပေးပါတယ်။ (Update အရမ်းများသွားရင် Model က စုတ်ပြတ်သွားတတ်လို့ပါ)။

- အားနည်းချက်: သူက ပုံမှန်အားဖြင့် Critic Model ဆိုတဲ့ နောက်ထပ် Model တစ်ခု ထပ်လိုပါတယ်။ ဒါကြောင့် Memory နဲ့ Compute အသုံးများပါတယ်။

#### ၂။ GRPO (Group Relative Policy Optimization)
ဒါကတော့ DeepSeek-V3 ဒါမှမဟုတ် DeepSeek-R1 တို့မှာ သုံးထားတဲ့ ပိုသစ်ပြီး ပိုထိရောက်တဲ့ နည်းလမ်းပါ။

- ဘယ်လိုအလုပ်လုပ်လဲ: PPO လို Critic Model အကြီးကြီး သုံးမယ့်အစား၊ Model ကနေ အဖြေအုပ်စု (Group) လိုက် ထုတ်ခိုင်းလိုက်ပါတယ်။ ပြီးရင် အဲဒီအုပ်စုထဲမှာတင် ဘယ်အဖြေက ပိုကောင်းလဲဆိုတာကို နှိုင်းယှဉ် (Relative) ပြီး အမှတ်ပေးတာပါ။

- ထူးခြားချက်: Critic Model မလိုတဲ့အတွက် Memory သက်သာပြီး တွက်ချက်ရတာ ပိုမြန်ပါတယ်။

- ဘာကြောင့် နာမည်ကြီးလဲ: AI ကို "စဉ်းစားပုံ စဉ်းစားနည်း" (Reasoning) သင်ပေးတဲ့နေရာမှာ အရမ်းထိရောက်တာကို တွေ့ရပါတယ်။



