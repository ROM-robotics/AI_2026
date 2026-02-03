# Reinforcement Learning with Neural Networks (StatQuest)
**Video URL:** https://www.youtube.com/watch?v=9hbQieQh7-o&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=23

## Claude summarized မြန်မာဘာသာ အကျဉ်းချုပ်

### အခြေခံအကြောင်းအရာ
ဒီ StatQuest video က Reinforcement Learning (RL) နဲ့ Neural Networks တွေကို ဘယ်လို တွဲသုံးကြောင်း ရှင်းပြထားပါတယ်။ RL ကို ဂိမ်းတွေ အနိုင်ရအောင်၊ ကားမောင်းတာ၊ ChatGPT ကို လူသားလို ပိုပြီး ပြောဆိုနိုင်အောင် လေ့ကျင့်တဲ့အခါ သုံးကြပါတယ်။

### ပင်မပြဿနာ
- သာမန် Neural Networks တွေကို လေ့ကျင့်တဲ့အခါ input data နဲ့ ideal output values တွေ လိုအပ်ပါတယ်
- ဒါပေမယ့် တချို့ပြဿနာတွေမှာ ကြိုတင်သိရှိထားတဃ့ output values တွေ မရှိပါဘူး
- ဥပမာ - စာမျက်နှာထဲက ဥပမာအရ အစာဆိုင် ၂ ခုကြားမှာ ဘယ်မှာ သွားစားရမယ်ဆိုတာ ကြိုမသိနိုင်ဘူး

### ဖြေရှင်းနည်း - Policy Gradients

#### 1. **Guess လုပ်ခြင်း**
- Output value ကို မသိရင် ခန့်မှန်းချက် (guess) လုပ်ပါတယ်
- ဥပမာ - Squatch's Fry Shack ကို သွားဖို့ ဆုံးဖြတ်ခဲ့ရင် အဲဒါက မှန်တယ်လို့ guess လုပ်တယ်

#### 2. **Derivative တွက်ခြင်း**
- အဲဒီ guess ကို အခြေခံပြီး derivative (အနုပညာ အခြေခံ အပြောင်းအလဲနှုန်း) ကို တွက်ချက်ပါတယ်
- Derivative က bias parameter ကို ဘယ်ဘက် သို့မဟုတ် ညာဘက် ရွှေ့ရမယ်ဆိုတာ ညွှန်ပြပေးပါတယ်

#### 3. **Reward သတ်မှတ်ခြင်း**
- အမှန်တကယ် လုပ်ဆောင်ပြီးတဲ့အခါ result ကို စစ်ဆေးပါတယ်
- မှန်ကန်တဲ့ ဆုံးဖြတ်ချက် ဖြစ်ရင် **positive reward** (+1.0) ပေးပါတယ်
- မှားယွင်းတဲ့ ဆုံးဖြတ်ချက် ဖြစ်ရင် **negative reward** (-1.0) ပေးပါတယ်

#### 4. **Updated Derivative တွက်ခြင်း**
```
Updated Derivative = Derivative × Reward
```
- Guess မှန်ရင် - derivative က မူလအတိုင်း အရပ်ညွှန်ပြပေးပါတယ်
- Guess မှားရင် - negative reward က derivative ကို ပြောင်းပြန်လှန်ပြီး မှန်ကန်တဲ့ ဘက်ကို ညွှန်ပြပေးပါတယ်

#### 5. **Parameter Update လုပ်ခြင်း**
- Gradient Descent ကို သုံးပြီး bias ကို update လုပ်ပါတယ်
```
New Bias = Old Bias - (Learning Rate × Updated Derivative)
```

### လက်တွေ့ ဥပမာ

**ပြဿနာ:** ဗိုက်ဆာနေပုံပေါ် မူတည်ပြီး အစာဆိုင် ၂ ခုကြားမှာ ရွေးချယ်ခြင်း
- **Input:** ဗိုက်ဆာနေမှု (0 = ဗိုက်မဆာဘူး, 1 = အရမ်းဆာတယ်)
- **Output:** Norm's Fry Hut သို့ သွားမည့် ဖြစ်နိုင်ခြေ (P_norm)

**လေ့ကျင့်မှု အဆင့်များ:**

1. **ပထမအကြိမ်:** ဗိုက်မဆာဘူး (0.0) → Squatch's သွားတယ် → အနည်းငယ် ရတယ် → မှန်တယ် → Reward = +1.0
2. **ဒုတိယအကြိမ်:** ဗိုက်မဆာဘူး (0.0) → Norm's သွားတယ် → အများကြီး ရတယ် → မှားတယ် → Reward = -1.0
3. အကြိမ်ပေါင်းများစွာ လေ့ကျင့်ပြီးတဲ့အခါ model က သင်ယူပါတယ်:
   - ဗိုက်မဆာရင် (0.0) → Squatch's သွားပါ (အနည်းငယ်ပဲ ရလေ့ရှိတယ်)
   - ဗိုက်ဆာရင် (1.0) → Norm's သွားပါ (အများကြီး ရလေ့ရှိတယ်)

### အဓိက အချက်များ

1. **Backpropagation ကွာခြားချက်:** 
   - သာမန် BP က ideal outputs လိုတယ်
   - RL က guesses နဲ့ rewards တွေ သုံးတယ်

2. **Policy Gradients အလုပ်လုပ်ပုံ:**
   - Guess → Calculate Derivative → Get Reward → Update Derivative → Update Parameters

3. **Reward System:**
   - Positive reward = မှန်ကန်တဲ့ ဆုံးဖြတ်ချက်
   - Negative reward = မှားယွင်းတဲ့ ဆုံးဖြတ်ချက်
   - Reward ပမာဏ များလေ step size ကြီးလေ

4. **အကျိုးသက်ရောက်မှု:**
   - Neural network က အကောင်းဆုံး policy (ဆုံးဖြတ်ချက်ချမှတ်ပုံ) ကို သင်ယူပါတယ်
   - Training data မှာ ideal outputs မပါလည်း လေ့ကျင့်နိုင်ပါတယ်

### လက်တွေ့အသုံးချမှုများ

- ဂိမ်း AI တွေ လေ့ကျင့်ခြင်း (AlphaGo, Chess AI)
- Self-driving cars
- Chatbots နဲ့ Language Models (ChatGPT)
- Robotics
- Resource optimization

### နိဂုံး

Reinforcement Learning က Neural Networks တွေကို ideal output values မရှိတဲ့ အခြေအနေမှာ လေ့ကျင့်နိုင်ဖို့ ခွင့်ပြုပေးပါတယ်။ Guess များ လုပ်ခြင်း၊ Rewards များ သတ်မှတ်ခြင်း၊ နဲ့ Derivatives များ update လုပ်ခြင်း ဆိုတဲ့ လုပ်ငန်းစဥ်များ ထပ်တလဲလဲ လုပ်ဆောင်ရင်း neural network က အကောင်းဆုံး behavior ကို သင်ယူသွားပါတယ်။

---

**Note:** အသေးစိတ် mathematical details တွေကို StatQuest ရဲ့ follow-up video မှာ လေ့လာနိုင်ပါတယ်။
