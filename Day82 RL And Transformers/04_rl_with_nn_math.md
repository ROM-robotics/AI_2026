### psa summarized
#### Neural Network and Reinforcement Learning Mathematics
- P(A) output နဲ့ P(A) Guess တို့ကို probability မို့လို့ distiance တွက်ဖို့ Cross Entropy တွက်ပါမယ်။ 
- ပြီးမှ Cross Entropy ကို Optimize လုပ်ဖို့ Bias နဲ့ Differentiate ပါမယ်။ 
- ဆိုတော့ 1 - P(A) = P(B) = P(Norm) ဆိုကြပါစို့။ P(A) = 1 - P(Norm) ပါ။
- အဲ့ဒီတော့ CE = -log(P(A)) = -log(1-P(Norm))

#### from sigmoid
- P(Norm) = sigmoid(x) = 1 / ( 1+e^x)
- x = input * weight + bias
- ဒါဆို ရှာရမှာက --> d CE p(A) / d Bias

#### chain rule သုံးပြီး အဆင့်လိုက် partial differentiate လုပ်ရပါမယ်။
- derivative =  differentiate error function respect to bias

#### updated derivative
- derivative * reward = updated_derivative
- step_size = learning_rate * updated_derivative
- new_bias = old_bias - step_size

#### plug new_bias to the NN
