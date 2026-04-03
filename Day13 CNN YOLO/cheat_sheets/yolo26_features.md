# YOLO26: Task-wise Explanation with Math Intuition

ဒီ note က workspace ထဲရှိ paper `2509.25164v5.pdf` ကိုအခြေခံပြီး YOLO26 ရဲ့ အဓိက ideas တွေကို Burmese နဲ့ ရှင်းပြထားတာပါ။

Paper အရ YOLO26 ဟာ Ultralytics YOLO family ထဲက edge-first design ကို ဦးစားပေးထားသော multi-task vision model ဖြစ်ပြီး အောက်ပါ tasks များကို support လုပ်ပါတယ်:

- Object Detection
- Instance Segmentation
- Pose / Keypoint Estimation
- Oriented Bounding Box Detection
- Classification

YOLO26 ရဲ့ core idea က accuracy တိုးအောင်ပဲမဟုတ်ဘဲ deployment လွယ်အောင် model graph ကိုပါ ရိုးရှင်းစေခြင်း ဖြစ်ပါတယ်။ Paper ထဲမှာ အဓိက highlight လုပ်ထားတာတွေက:

- DFL ကို ဖြုတ်ထားခြင်း
- End-to-end NMS-free inference
- ProgLoss
- STAL
- MuSGD optimizer

## 1. YOLO26 ကို ဘာကြောင့် special လို့ခေါ်နိုင်သလဲ

YOLO26 မှာ previous YOLO versions တွေထက် ပိုသိသာတဲ့ architectural change နှစ်ခုရှိပါတယ်:

### 1.1 DFL ကို ဖယ်ရှားထားခြင်း

အရင် YOLO models တချို့မှာ bounding box coordinate ကို တိုက်ရိုက်မထုတ်ဘဲ distribution တစ်ခုအဖြစ်ခန့်မှန်းပြီးမှ expectation ယူသလို regression လုပ်ကြပါတယ်။ ဒါကို Distribution Focal Loss (DFL) လို့ခေါ်ပါတယ်။

Math intuition:

အရင် idea က coordinate တစ်ခုကို scalar တန်ဖိုးတစ်ခုအနေနဲ့ မထုတ်ဘဲ bin probability distribution အဖြစ်ခန့်မှန်းသည်ဟုယူလို့ရပါတယ်။ ဥပမာ x-coordinate အတွက်

$$
p(x=i),\quad i \in \{0,1,2,\dots,K-1\}
$$

ပြီးမှ expected value နဲ့ coordinate ပြန်ယူမယ်:

$$
\hat{x} = \sum_{i=0}^{K-1} i \cdot p(x=i)
$$

ဒီနည်းက localization ပိုနူးညံ့နိုင်ပေမယ့် inference graph ပိုရှုပ်စေတယ်။ YOLO26 က ဒီ step ကိုဖြုတ်လိုက်တာကြောင့် coordinate ကို တိုက်ရိုက် regression လုပ်တဲ့ဘက်ကိုပြန်သွားတယ်လို့ နားလည်နိုင်ပါတယ်:

$$
\hat{b} = (\hat{c_x}, \hat{c_y}, \hat{w}, \hat{h})
$$

အကျိုးကျေးဇူးက:

- inference မြန်လာတယ်
- export လုပ်ရတာ ပိုလွယ်လာတယ်
- ONNX / TensorRT / CoreML / TFLite ပေါ်မှာ ပို hardware-friendly ဖြစ်လာတယ်

### 1.2 NMS-free inference

Traditional detector တွေမှာ object တစ်ခုအတွက် box အများကြီးထွက်နိုင်လို့ post-processing အဖြစ် NMS သုံးရပါတယ်။

NMS intuition က overlap များတဲ့ boxes တွေထဲက confidence အမြင့်ဆုံးတစ်ခုကိုပဲ ထားပြီး ကျန်တာကို ဖယ်ပစ်တာဖြစ်တယ်။

IoU ကို:

$$
\mathrm{IoU}(A,B) = \frac{|A \cap B|}{|A \cup B|}
$$

လို့ရေးနိုင်ပြီး overlap များလွန်းတဲ့ boxes တွေကို suppress လုပ်တယ်။

YOLO26 က paper အရ prediction head ကို end-to-end ပုံစံနဲ့ redesign လုပ်ထားပြီး final detections ကို model က တိုက်ရိုက်ထုတ်ပေးချင်တဲ့ direction ကိုသွားပါတယ်။ အဓိပ္ပာယ်က deployment time မှာ extra filtering logic ကို နည်းသွားစေတယ်။

Practical intuition:

- အရင်ပုံစံ: model က candidate boxes အများကြီးထုတ်, နောက်မှ NMS နဲ့ရှင်း
- YOLO26 ပုံစံ: model ကို duplicate နည်းတဲ့ final predictions ကို တိုက်ရိုက်ထုတ်နိုင်အောင် train လုပ်

ဒါကြောင့် latency လျော့တယ်၊ threshold tuning complexity လျော့တယ်။

## 2. YOLO26 Unified Pipeline

Task မတူပေမယ့် shared backbone/neck feature extractor ကိုသုံးပြီး head များသာ ပြောင်းသွားတယ်လို့ နားလည်ရင် လွယ်ပါတယ်။

အကြမ်းဖျဉ်းအားဖြင့်:

$$
x \xrightarrow{\text{Backbone}} F \xrightarrow{\text{Neck}} \{P_3, P_4, P_5\} \xrightarrow{\text{Task Head}} y
$$

ဒီမှာ:

- $x$ = input image
- $F$ = extracted semantic features
- $P_3, P_4, P_5$ = multi-scale features
- $y$ = task-specific outputs

Multi-scale feature maps သုံးရတဲ့ intuition က object size မတူတာကြောင့်ဖြစ်တယ်။

- small object တွေကို high-resolution feature maps က ကောင်းကောင်းဖမ်းနိုင်တယ်
- large object တွေကို low-resolution but semantically-rich feature maps က ကောင်းကောင်းဖမ်းနိုင်တယ်

## 3. Detection

Detection မှာ model ရဲ့ target က object ရှိတဲ့နေရာနဲ့ class ကို ခန့်မှန်းခြင်း ဖြစ်တယ်။

### 3.1 Output form

Object တစ်ခုစီအတွက် prediction ကို အောက်ပါပုံစံနဲ့စဉ်းစားလို့ရတယ်:

$$
\hat{y} = (\hat{b}, \hat{p})
$$

ဒီမှာ:

- $\hat{b} = (\hat{c_x}, \hat{c_y}, \hat{w}, \hat{h})$
- $\hat{p}$ = class probabilities

Class probability ကို sigmoid/softmax type output ကနေယူတတ်တယ်:

$$
\hat{p}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

### 3.2 Math intuition

Detection ကို geometry + classification ပေါင်းထားတဲ့ problem လို့မြင်ရင်လွယ်တယ်:

- geometry side: box position မှန်ရမယ်
- semantics side: class မှန်ရမယ်

Loss ကို conceptually:

$$
L_{det} = \lambda_{box} L_{box} + \lambda_{cls} L_{cls}
$$

လိုမျိုး စဉ်းစားနိုင်တယ်။

$L_{box}$ အတွက် IoU-based loss သို့မဟုတ် CIoU/GIoU style loss များကိုသုံးလေ့ရှိပြီး၊ intuition က predicted box နဲ့ ground truth box overlap အများဆုံးဖြစ်ဖို့ဖြစ်တယ်။

ဥပမာ simplest form:

$$
L_{box} \approx 1 - \mathrm{IoU}(\hat{b}, b)
$$

ဒီ loss က 0 နီးလာလေလေ box က ground truth နဲ့ပိုတိကျလာတယ်။

### 3.3 YOLO26-specific intuition

YOLO26 မှာ DFL မရှိတော့တာကြောင့် box regression path ပိုတိုလာတယ်။ NMS-free inference ကြောင့် output pipeline ကလည်း ပိုရှင်းသွားတယ်။ Paper ထဲက benchmark အရ detection performance က `YOLO26x` မှာ COCO mAP50-95 = 57.5 ထိရတယ်။

## 4. Instance Segmentation

Detection က box အတိုင်းအတာတင်မကဘဲ object ရဲ့ pixel-wise shape ကိုပါ ခွဲချင်ရင် instance segmentation လိုလာတယ်။

### 4.1 Output form

Prediction ကို:

$$
\hat{y} = (\hat{b}, \hat{p}, \hat{M})
$$

ဒီမှာ $\hat{M}$ က object mask ဖြစ်တယ်။

Ultralytics-style segmentation intuition အရ shared feature maps ပေါ်က prototype masks တွေနဲ့ object-specific coefficients တွေကိုပေါင်းပြီး final mask ဆောက်တယ်လို့မြင်ရင်လွယ်တယ်:

$$
\hat{M}(u,v) = \sigma\left(\sum_{k=1}^{K} \alpha_k P_k(u,v)\right)
$$

ဒီမှာ:

- $P_k(u,v)$ = prototype mask basis
- $\alpha_k$ = object-specific coefficients
- $\sigma$ = sigmoid

### 4.2 Math intuition

ဒီနည်းက linear combination of basis functions လိုစဉ်းစားလို့ရတယ်။ Object တစ်ခုချင်းအတွက် full mask image တစ်ခုလုံးကိုအသစ်ထုတ်တာမဟုတ်ဘဲ shared mask patterns တွေကို coefficients နဲ့ပေါင်းပြီး object shape ဆောက်တာဖြစ်တယ်။

အဲဒါကြောင့်:

- computation သက်သာတယ်
- feature sharing ကောင်းတယ်
- detection နဲ့ segmentation ကို joint learning လုပ်လို့ရတယ်

Loss ကို conceptually:

$$
L_{seg} = L_{det} + \lambda_{mask} L_{mask}
$$

mask loss က BCE, Dice, သို့မဟုတ် overlap-based loss ဖြစ်နိုင်တယ်။

Paper ထဲက metric အရ `YOLO26x-seg` က box mAP50-95 = 56.5, mask mAP50-95 = 47.0 ရရှိတယ်။

## 5. Pose / Keypoint Estimation

Pose estimation မှာ object ကို box နဲ့တွေ့ရတာတင်မက body joints သို့မဟုတ် landmark points တွေကိုပါ ခန့်မှန်းရတယ်။

### 5.1 Output form

Object တစ်ခုအတွက် keypoints $J$ ခုရှိတယ်ဆိုပါစို့:

$$
\hat{K} = \{(\hat{x}_1, \hat{y}_1), (\hat{x}_2, \hat{y}_2), \dots, (\hat{x}_J, \hat{y}_J)\}
$$

visibility/confidence ပါထည့်ရင်:

$$
\hat{K}_j = (\hat{x}_j, \hat{y}_j, \hat{v}_j)
$$

### 5.2 Math intuition

Pose estimation ကို "where are the important joints?" ဆိုတဲ့ structured regression problem လို့မြင်နိုင်တယ်။

Loss ကို conceptually:

$$
L_{pose} = L_{det} + \lambda_{kpt} \sum_{j=1}^{J} v_j \cdot \ell\big((\hat{x}_j,\hat{y}_j), (x_j,y_j)\big)
$$

ဒီမှာ:

- $v_j$ = ground-truth visibility
- $\ell$ = L1 / SmoothL1 / normalized keypoint loss တစ်မျိုးမျိုး

Visibility weighting ထည့်တာက မမြင်ရတဲ့ joint ကို over-penalize မလုပ်စေချင်လို့ပါ။

Geometric intuition:

- detection က object "ဘယ်မှာရှိလဲ" ကိုပြောတယ်
- pose က object ရဲ့ internal structure "အတွင်းပိုင်း landmarks ဘယ်လိုစီထားလဲ" ကိုပြောတယ်

Paper ထဲက pose benchmark အရ `YOLO26x-pose` က COCO pose mAP50-95 = 71.6 ထိရတယ်။

## 6. Oriented Bounding Box Detection

သာမန် detection box က axis-aligned rectangle ဖြစ်တယ်။ ဒါပေမယ့် drone imagery, satellite imagery, text blocks, ships, vehicles လို အရာတွေက လှည့်ထားနိုင်တယ်။ ဒီအခါ rotated box လိုလာတယ်။

### 6.1 Output form

Oriented box ကို:

$$
\hat{b}_{obb} = (\hat{c_x}, \hat{c_y}, \hat{w}, \hat{h}, \hat{\theta})
$$

လို့ရေးနိုင်တယ်။ ဒီမှာ $\theta$ က rotation angle ဖြစ်တယ်။

Corner points တွေကို rotation matrix နဲ့ ရနိုင်တယ်:

$$
R(\theta) =
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
$$

Box local corners $q_i$ တွေကို center ထဲကနေ rotate လုပ်ပြီး image coordinates ထဲသို့ပြောင်းတယ်:

$$
p_i =
\begin{bmatrix}
c_x \\
c_y
\end{bmatrix}
+ R(\theta) q_i
$$

### 6.2 Math intuition

Axis-aligned box က object ကိုပတ်ရင် background အလွန်များနိုင်တယ်။ Rotated box သုံးလိုက်ရင် object orientation ကိုလိုက်ဖက်ပြီး tight localization ရတယ်။ အထူးသဖြင့် aerial imagery မှာ road direction, ship heading, building orientation စတာတွေကို ပိုကောင်းကောင်း represent လုပ်နိုင်တယ်။

Loss က rotated IoU သို့မဟုတ် angle-aware box regression concept ကို အခြေခံတတ်တယ်:

$$
L_{obb} = \lambda_{rbox} L_{rbox} + \lambda_{cls} L_{cls}
$$

Paper အရ `YOLO26x-obb` က DOTA v1 ပေါ်မှာ mAP50-95 = 56.7 ရရှိတယ်။

## 7. Classification

Classification မှာ localization မလိုဘဲ image တစ်ပုံလုံးကို class တစ်ခု သို့မဟုတ် label တစ်ခု assign လုပ်တယ်။

### 7.1 Output form

Backbone feature $f$ ကနေ logits vector $z$ ထုတ်တယ်:

$$
z = Wf + b
$$

ပြီးတော့ probabilities ကို softmax နဲ့ယူမယ်:

$$
p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}
$$

Prediction class က:

$$
\hat{c} = \arg\max_k p_k
$$

### 7.2 Math intuition

Classification က feature space ထဲမှာ class boundary ဆွဲတဲ့ problem လို့မြင်နိုင်တယ်။ Backbone က image ကို feature vector တစ်ခုအဖြစ် compress လုပ်တယ်။ Head က အဲဒီ vector ဟာ class ဘယ်ဘက်ကိုပိုနီးလဲဆိုတာ linear scores နဲ့စစ်တယ်။

Loss က cross entropy:

$$
L_{cls} = -\sum_k y_k \log p_k
$$

ဒီမှာ $y_k$ က one-hot ground truth ဖြစ်တယ်။

Paper ထဲမှာ `YOLO26x-cls` က ImageNet Top-1 accuracy = 79.9%, Top-5 = 95.0% ရရှိတယ်။

## 8. ProgLoss, STAL, MuSGD တို့ရဲ့ Math Intuition

## 8.1 ProgLoss

Loss terms များစွာရှိတဲ့ model တစ်ခုမှာ training အစပိုင်းနဲ့ အဆုံးပိုင်းမှာ important objective မတူနိုင်ဘူး။ ProgLoss က loss weights ကို static မထားဘဲ epoch အလိုက် dynamic ပြောင်းတယ်လို့ နားလည်လို့ရတယ်:

$$
L = \sum_i \lambda_i(t) L_i
$$

ဒီမှာ $\lambda_i(t)$ က training step/epoch $t$ ပေါ်မူတည်ပြီး ပြောင်းတယ်။

Intuition:

- training အစပိုင်း: coarse localization, easy patterns, stable gradients
- training နောက်ပိုင်း: harder cases, rare classes, fine localization

ဒါကြောင့် task တစ်ခုတည်းက dominating မဖြစ်ဘဲ learning balance ပိုကောင်းလာတယ်။

## 8.2 STAL

Small objects တွေက pixel အနည်းငယ်ပဲရှိလို့ assignment step မှာ background နဲ့ လွယ်လွယ်ပျောက်နိုင်တယ်။ STAL က small targets တွေကို positive assignment ပိုသေချာအောင်လုပ်တဲ့ idea လို့ယူလို့ရတယ်။

Intuition အရ label assignment score ကို:

$$
s = s_{cls} \cdot s_{loc} \cdot s_{size}
$$

လိုမျိုးမြင်နိုင်ပြီး $s_{size}$ term က tiny objects အတွက် priority boost ပေးတဲ့သဘောဖြစ်တယ်။ Paper က exact formula မဖော်ပြပေမယ့် main idea က "small object ကို training signal ပိုမပျောက်စေချင်ခြင်း" ဖြစ်တယ်။

## 8.3 MuSGD

Optimizer update ကို general form နဲ့ရေးရင်:

$$
	heta_{t+1} = \theta_t - \eta_t \cdot g_t
$$

ဒီမှာ $g_t = \nabla_\theta L(\theta_t)$ ဖြစ်တယ်။

SGD က generalization ကောင်းတတ်ပေမယ့် convergence ပိုနှေးနိုင်တယ်။ Adaptive method တွေက မြန်တတ်ပေမယ့် overfit သို့မဟုတ် unstable ဖြစ်တတ်တယ်။ MuSGD ဆိုတာ paper အရ SGD + Muon-inspired behavior ကို hybrid လုပ်ထားတဲ့ optimizer ဖြစ်ပြီး intuition က:

- SGD ရဲ့ stable generalization ကိုယူ
- adaptive curvature/momentum-like benefit ကိုယူ
- training ကို ပိုချောမွေ့စေ

ဒါကြောင့် fewer epochs နဲ့ plateau ကောင်းကောင်းရောက်နိုင်တယ်လို့ paper ကဆိုတယ်။

## 9. Task တစ်ခုချင်းစီကို ခွဲပြီးလွယ်လွယ်မှတ်မယ်ဆိုရင်

### Detection

"Object ဘယ်မှာရှိလဲ, ဘာ class လဲ" ကို box + class probability နဲ့ခန့်မှန်းတာ။

### Instance Segmentation

"Object ဘယ်မှာရှိလဲ" အပြင် "pixel တိတိကျကျ ဘယ် shape လဲ" ကိုပါ mask နဲ့ထုတ်တာ။

### Pose / Keypoint

"Object အတွင်းမှာ joints / landmarks တွေ ဘယ်နေရာရှိလဲ" ကို point set အနေနဲ့ခန့်မှန်းတာ။

### Oriented Box

"Object ဘယ်မှာရှိလဲ" အပြင် "ဘယ် angle နဲ့လှည့်ထားလဲ" ကိုပါတွက်တာ။

### Classification

"ဒီပုံက ဘာအမျိုးအစားလဲ" ကို image-level label တစ်ခုအနေနဲ့ထုတ်တာ။

## 10. YOLO26 ကို conceptually တစ်ကြောင်းနဲ့ ပြောရရင်

YOLO26 က YOLO family ကို "ပိုရှုပ်အောင် feature ထပ်ထည့်" တဲ့လမ်းမဟုတ်ဘဲ "predictor ကို ရိုးရှင်းစေ, duplicate filtering ကို model ထဲသို့ပေါင်းထည့်, training signal ကို rebalance လုပ်, small objects ကို မပျောက်စေ, deployment ကို hardware-friendly လုပ်" ဆိုတဲ့လမ်းနဲ့ တိုးတက်စေထားတဲ့ version လို့ ပြောနိုင်ပါတယ်။

## 11. Practical takeaway

- Detection အတွက် YOLO26 က direct box regression + NMS-free path ကိုဦးစားပေးတယ်
- Segmentation မှာ shared features ပေါ်က lightweight mask branch ကိုသုံးတယ်
- Pose မှာ object landmarks ကို structured regression အဖြစ်ဖြေရှင်းတယ်
- OBB မှာ angle ပါသော geometry ကို model လုပ်တယ်
- Classification မှာ compact feature-to-logit mapping နဲ့ efficient recognition လုပ်တယ်
- Deployment အတွက် DFL removal + NMS-free design က export/quantization ကိုလွယ်စေတယ်

## 12. Important note

Paper က benchmarking and architectural review အမျိုးအစားဖြစ်လို့ head implementation detail အတိအကျ, exact matching strategy, exact closed-form loss equations အားလုံးကို source code level အထိမဖော်ပြထားပါဘူး။ ဒါကြောင့် အထက်က math sections တွေဟာ paper claims နဲ့ Ultralytics-style YOLO design intuition ကိုပေါင်းပြီး နားလည်လွယ်အောင် ရေးထားခြင်းဖြစ်ပါတယ်။
