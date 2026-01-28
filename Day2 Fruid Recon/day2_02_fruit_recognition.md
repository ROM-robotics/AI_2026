## Day 2
- some python modules , os modules
- just theory 
- Fruit Recognition system
- Face Recognition system without NN, CNN
 
### FRUIT RECOGNITION
- [photos](https://drive.google.com/drive/folders/1_A0ldPMhr0CSwZLrDjxnFJD4Goq1lDfh?usp=sharing)
- [fruit dataset](https://drive.google.com/file/d/1LKqGV-k_kZBhe_8it3h-0y8byIXsLkTi/view?usp=sharing)
- [lecture](https://drive.google.com/file/d/1hprbV6fZdzoEysM5mP2E8RxqlWLCzrYA/view?usp=sharing)
- [code](https://www.google.com)

#### HOG 01 to 07
- Historgram of Oriented and Gradient
- matrix နဲ့ တိုက်စစ်
- pixel တခုကို magnitude, orientation တွက်
- vector 9 orientation ကို ကြည့်ပြီး magnitude တွေခွဲထည့်
- ပြီးပေါင်းလိုက် အဲ့တာ ကို feature vector ယူလိုက်တာ
- ပြီးတိုက်စစ် convolute လုပ်တာ

#### photos 13, 14, 15
- ရလာဒ် ကို ကြည့်
- image သွင်းပြီး data ထွက်လာတယ်လို့ စပြောလို့ရပြီ။

#### photo 16
- fruit recognize အသေး, color feature only နဲ့ မရ , edge feature ပါထည့်
- cnn လောက်ဆို ဘာ feature လည်း ပြောမရတော့ အကုန် random က စတင်တယ်
- ကြည့်ရှင်းပါ

#### IMAGE ACQUISITION 1
- source code နေရာ, ပုံ 20000 လောက် 
- apple directory 5 ခု

#### photo 2
- ဒါက image aciquisition လုပ်ပုံ, scikit-image
- glob module နဲ့ LoadImage
- apple directory ၅ခု , ပုံ 3098 ကျော် -> numpy array
- ပုံတပုံ 2 dimensional numpy array, အဲ့လိုပုံ 3098, 3 dimension
- label လည်း 3098 ကျော်
- spyder မှာ ပုံတပုံ row 1 ခု , size မတူ

#### photo 3
- spyder မှာ data frame ကြည့်ပုံ
- ပုံတပုံ dimension 3 ခု 3 channel
- size တွေ သိပ်မတူ, crop, resize မလုပ်ရသေး

#### photos 4, 5
- ပုံတပုံချင်းစီကြည့်ရင်တော့ ဒါမျိုးတွေ့ရ , row column channel သိဖို့လို
- ဘယ် axis ကကြည့်နေလဲ သိဖို့လို , axis အယူအဆနားလည်ဖို့လို
- axis change ပြီးကြည့်တတ်ဖို့လို

#### photo 6
- ဒီ cube လိုပဲလေ ရှေ့တည့်တည့်ကကြည့်ရင် အဖြူဘဲမြင်ရ

#### photo 7
- ဒီမှာဆို folder တွေကို iteration လုပ်ပြီး image matrix တနည်းအားဖြင့် numpy array တွေရော , label တွေရော list ထဲထည့်
- nympy array image နဲ့ label တွေကို data frame ဆောက်ထားတယ်
- shape(0) နဲ့ ပုံပေါင်း ၃၀၉၈ ကိုထည့်လိုက်တယ်။

#### photo 8
- ဒါက imshow နဲ့ data acquisition လုပ်ထားတာတွေကို ထုတ်ကြည့်တာ
- ဒါသည် preprocessing မလုပ်မီ data acquisition လုပ်တာ data frame ဆောက်တယ်။ ဘာမှမလုပ်ရသေး။

#### PREPROCESSING
- preprocessing မလုပ်မီ train test split လုပ်ဖို့လို

#### photo 2
- ဒီတခါ pandas နဲ့ မလုပ် , concatenate တိုက်ရိုက်လုပ် 
- ပုံပေါင်း 19127 , label ပေါင်း 19127

#### photo 3
- testing size 0.33, random, shuffle

#### photo 4
- အခု train test split လုပ်ပြီးပြီဆိုတော့ ml model, classifier တစ်ခုသုံးပြီး စမ်းလို့ရပြီ
- svm, knn ( neighour မကိန်း ) လိုကောင်မျိုးပေါ့
- အခု pricipal component analysis ပြောမယ်။ 
- ဒါဆို ml algorithm အနေနဲ့ knn, pca ပြောပြီးပြီပေါ့

#### photo 5
- အဲ့ဒါမပြောခင် အခု preprocessing လုပ်ဖို့လို
- preprocessing က ဘာလုပ်တာလဲ?
- crop, roi bg ခွဲထုတ်, resize လုပ်မယ်
- postprocessing က feature ဆွဲထုတ်လို့ရအောင် dilate, erode, connected component ဖြစ်အောင်လုပ်

#### photo 6
- အခု ဒီမှာတေ့ preprocessing အနေနဲ့ ဘာမှအထူတလည် မလုပ်
- grayscale ပြောင်း , resize လုပ် ဒါပဲလုပ်ထားတယ်

#### FEATURE EXTRACTION
- preprocessing ပြီးတဲ့အခါ feature ဆွဲထုတ်မယ်။
- ရှေ့တုန်းကလို color feature, edge feature ကို့ဟာကိုယ် မထုတ်နဲ့ နားလည်အောင်ပြတာ, တိုင်ပတ်သွားမယ်။

#### photo 2
- HOG ကိုသုံးပြီး feature ဆွဲထုတ်မယ်။
- iteration လုပ်ပြီး hog လုပ်ကာ array အနေနဲ့ return ပြန်
- hog orientation 9, pixel per cel 8

#### photo 3
- ဒါက scikit-image မှာ ခုနကလို processing တွေလုပ်ချင်တဲ့အခါ api ref လာကြည့်လို့ရတယ်။

#### photo 4
- feature ထဲမှာ hog gradient ထုတ်

#### photo 5,6
- hog method လေ့လာကြည့်
- parameters, argument တွေလေ့လာကြည့်

#### photo 7, 8
- testing အတွက်လည်း training ပြီးရင် စမ်းသပ်ရမှာမို့လို့ တခါထဲ fv ဆွဲထုတ်ထားတယ်။

#### CREATE MODEL AND TESTING
- ဒါဆို လွယ်တဲ့ အပိုင်းကိုရောက်ပြီး ml model တခုဖြစ်တဲ့ K neighbours classifier ကို သုံးရုံပဲ။
- scikit-learn ထဲက metric ( တိုင်းတာခြင်း ) ထဲက accuracy score သုံးမယ်။
#### photo 2
- ဒါက knn အတွက် parameters တွေ, knn ပြောတုန်းက 

#### photo 3
- neighbours အရေအတွက် မကိန်း, အခု ၁၁ ခု
- train မယ်ဆိုတော့ fit method နဲ့ input data ထည့်မယ်။ အဖြေမှန်တွေဖြစ်တဲ့ ground truth or label ထည့်မယ်
- နောက် testing data input ထည့်ပြီး output ဖြစ်တဲ့ y hat , y_predict ရလာပါမယ်။
- ရတဲ့ အဖြေဘယ်လောက်မှန်လဲသိရအောင် y hat နဲ့ ground truth ဖြစ်တဲ့ y တို့ကို accuracy တွက်ခိုင်းပါတယ်။
- ဒီမှာဆိုရင် accuracy 100 ရတယ် ဒါပေမဲ့ကောင်းတယ်လို့ပြောလို့မရဘူး
- တို့တွေ လိုချင်တာက precision, recall စတာတွေဖြစ်နိုင်တာမို့လို့ပါ။

#### photo 4, 5
- ဒါကတော့ ရတဲ့ အဖြေတွေကို label ထိုးပြီး imshow နဲ့ ပြခိုင်းတာ
- ရလာတဲ့ အထဲက ပုံ ၁၅ ခုကို label ထိုးပြတာပါပဲ


#### PICKLE MODULE
- ရလာတဲ့ ml model ကို save ပြီး ပြန်သုံးလို့ရတယ်။
- pickle module သုံးပြီး dump နဲ့ save လို့ရတယ်။ အဲ့ဒီ fv တွေပါတဲ့ matrix ကြီးကို disk ပေါ် save လိုက်တာပါပဲ။
- ဒါကို မမှတ်ပါနဲ့ နောက်ပိုင်းကျ cnn, llm တွေနဲ့ သွားတဲ့အခါ အခြား model သိမ်းတဲ့ နည်းတွေရှိပါတယ်။
- wb နဲ့ write byte


#### photo 2
- ပြန်သုံးချင်ရင် ဒီမှာ save ထားတဲ့ကောင်ပါ
- ဒါပေမဲ့ ဒီနေ့ခေတ်မှာ ဒီနည်းမသုံးနဲ့တော့ နောက်ပိုင်း tensor, torch နဲ့ သွားရင် high level technique တွေရှိ

#### photo 3
- ဒီမှာ model ကို open နဲ့ ဖွင့်ပြီးတော့ load လုပ်ပြီးသုံးတယ်။
- testing data ကို ထပ်စမ်းသပ်ပုံကို ပြထားတယ်။
- အိုကေ မေးခွန်းတစ်ခုမေးမယ်
- တို့ မှာ သစ်သီးပုံတစ်ပုံရှိတယ်ဆိုပါစို့ ။ အခု ဖန်တီးထားတဲ့ ml model ကို သုံးလို့ရမလား။ ဘာပုံလဲလို့
- ရပါတယ်။ 
- ဒါဆို code ဘယ်လိုရေးမလဲ, code ရေးမည့် logic ပြောပြပါ။


 



















































