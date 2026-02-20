# Naive Bayes: Quick Guide & 5 Implementation Variants

ဒီ README မှာ Naive Bayes Classifier အမျိုးအစား (၅) မျိုးကို ဘယ်အချိန်မှာသုံးရမလဲဆိုတာနဲ့ Preprocessing/Testing နည်းလမ်းတွေကို လိုတိုရှင်း စုစည်းဖော်ပြထားပါတယ်။

---

## 1. Gaussian Naive Bayes
* **Best for:** Continuous (numeric) features (ဥပမာ- Iris dataset, sensor readings)။
* **Note:** Data သည် Normal Distribution မဖြစ်ပါက Performance ကျနိုင်သဖြင့် Feature Transformation ကို စဉ်းစားပါ။

## 2. Multinomial Naive Bayes
* **Best for:** Discrete counts/frequencies (ဥပမာ- စာသားထဲရှိ Word counts)။
* **Note:** Sparse Matrix များကို ကိုင်တွယ်ရာတွင် အလွန်ထိရောက်သည်။

## 3. Bernoulli Naive Bayes
* **Best for:** Binary features (0 သို့မဟုတ် 1)။
* **Note:** စာသားတိုများ သို့မဟုတ် Feature ရှိမရှိကသာ အဓိကကျသော Task များတွင် သုံးသည်။

## 4. Complement Naive Bayes
* **Best for:** Imbalanced multi-class text data။
* **Note:** Multinomial တွင်ဖြစ်တတ်သော Bias ပြဿနာကို ဖြေရှင်းရန် အကောင်းဆုံးဖြစ်သည်။

## 5. Categorical Naive Bayes
* **Best for:** Categorical (Nominal) features များ။
* **Note:** Continuous data မဟုတ်သော Category သီးသန့် feature များအတွက် ထိရောက်သည်။

---

### Comparison Summary
| Model | Feature Type | Main Use Case |
| :--- | :--- | :--- |
| **Gaussian** | Real Numbers | Physical measurements |
| **Multinomial** | Word Counts | Text classification |
| **Bernoulli** | Binary (0/1) | Spam detection (presence) |
| **Complement** | Counts (Imbalanced) | Skewed text datasets |
| **Categorical** | Categories | Survey data/Nominal features |