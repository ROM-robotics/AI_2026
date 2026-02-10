existing challenges that people are dealing with in industry. Uh so first topic we want to talk about is um I know
RL Theory vs Applications
a lot of times when people are learning RL there's a lot of theory but then when people try to apply that in application
sometimes it could be quite different from theory. So, uh, if you were to give the audience like a good overview of
what that difference is, how would you describe that? Uh, yeah. So, I think having a very
solid background and foundation in RL theory is actually very rewarding and
inspiring for RL practice. So um I think RL is more than just collecting rewards
but also like how do we balance the exploration and exploitation uh that is
actually embedded in the coefficients and the uh exploration uh re uh
exploration formulation in the RL theory itself. So how do we uh gain some
insights from uh tuning the RO rewards or or coefficient from the RO theory I
think is a very uh interesting and uh rewarding topic. Okay. So if someone is like for example
if they're doing the tuning with all the coefficients is there like a specific
example where you could think of where you know something where they try to apply that maybe in um like simulation
but then when they actually try to deploy it it's very different. uh do are you referring to the sim to
row gap or just uh how is it what is a systematic way to tune the RL
coefficients and reward uh in sim? Yeah, I guess more so in terms of the
theory like um you know because a lot of times when someone if someone were to jump in and use a framework like um
gymnasium for example, they have everything set up where a lot of the function calls is kind of like a black
box, right? they would maybe play with the reward structure or maybe they might
change kind of like the exploration you were saying. So, you know, for example, someone who's very used to a gymnasium
Python framework and um those that's what I would maybe consider more of the application side. So where do you see
that that infrastructure start to break where you know like for someone who has
spent maybe a couple years learning the theory what is something what edge do
someone that know the theory might have over someone who doesn't you know if they're just playing with gymnasium for
example um yeah that's a great question so uh during the training we might see some
phenomenon as mode collapse. So for example uh if there is not enough enough
exploration from the agents the agents will just quickly uh decay to a single
behavior without exploring the environment further and it can easily get stuck in the local minimum which uh
we don't want as the ideal behavior. So in that case uh we might one might have
to increase the exploration coefficient to encourage the agent to explore the
environments before collapsing to a single mode and doing explor exploitation afterwards.
Okay. So someone who's just playing with the gymnasium um like Python library for
example is that something they would be able to
figure out or is it because if they had like no theory they wouldn't be able to figure out something like that
I think if you yourself do enough of trial and error and change the coefficients this and there you might be
able to bump into some uh coefficients tuning or uh you uh you just do a grid
search on all the important co uh coefficients and get a very good intuition afterwards uh like which part
which are the most important coefficients or parameters you should change for the RL training but if we
just start from the fundamental of the RO training and RO theory itself so most
of the time the very common algorithm we're using for RO these days are PO so
there are only a small number of terms in the PO that are really important
that is causing the training to uh stabilize or destabilize. So if we just start from these formulation uh from the
fundamental we will be have able to have like a big picture of oh what are the important parameters we should tune.
So something like doing like a grid search on all the coefficients. Do you feel like that's more of a advanced
technique that someone would learn maybe only through school or is that something that someone who's more familiar with
the application only part would also know how to do? Uh I think people who are very practical
will also come up with this idea. Oh really? Okay. Um you mentioned the PO. I know that's
one of the most common uh RL models that people are using right now for robotics.
Um would you say there's a specific reason why people are leaning towards that versus another model or how they
come to decide to use that one? Yeah. Um so uh in Local Motion for example, people use PO a lot because
it's uh an online uh policy that is uh very good at encouraging the uh
encouraging the training to stay within the distribution. So that means the uh
RL algorithm is actually experiencing the state the the same state and action distribution as the agent itself is
exploring the environment. So for some other uh more efficient RL algorithms
such as offline RL uh they are essentially using different strategies uh during the policy update versus what
the agent is using to explore the environment. That kind of algorithm is
uh cheaper and is can can be more uh efficient but because the policy update
and the agent is experiencing different state distribution uh during deployment there will be some non ideal effect
called distribution shift which can cause a a big gap during deployment and
training. So would you say if you were to give like a percentage of users
that's using PO versus other models, what would how would you kind of describe that distribution right now?
Uh I would say probably 90% of robot locomotion researchers are using PO.
Sim-to-real gap for RL Deployment
Oh okay. We were talking about like the sim to real gap. That's like a very big area that people are trying to tackle
and I know there's people that try to tackle that problem either you know rand randomizing
uh you know physical properties of their robots or some people have like zeroot
techniques and also there's people that try to train on hardware ignoring some of the simulation. in your opinion, what
is the best approach to sim to real? And maybe if you could explain some of the tradeoffs between uh maybe some of the
methods I mentioned or any other methods that um you're familiar with.
Yeah, that's a great question. Simple is a very important procedure in our uh
entire robot deployment pipeline. So I would say we one should start with a
reasonably accurate system identification of the robot and then to
try to randomize a little bit around that nominal values in simulation during training. So that's called domain
randomization uh and deployment uh and and deploy it on the robots and see if
there's other like mismatch between the sim and row. So one can essentially do
uh a loop where uh one identify the coefficient system coefficients of the
robots uh try to uh randomize it in sim deploy it on real and collect the system
rowouts uh to back prop the parameters back to the sim to modify the sim
parameters to match the real behavior. Uh that would be the most ideal case. uh but also one can uh train a base policy
in simulation which is performing reasonably enough and then starting from that base policy uh do real world RL uh
in real uh I would say training RL from scratch in the real world is very uh
time consuming and also can uh do a lot of damage to the hardware which is the
hardware can be expensive so we want to u minimize the uh computation time as
well as the cost for the entire training and deployment loop. Yeah, that's a good point. Uh on the
hardware note, I'm very curious um like have you ever encountered any issues
where um your hardware randomly fails during your RL deployment and you
couldn't figure it out or maybe something that was harder hard for you to figure out.
Uh yeah. So actually that's a very common issue when we're doing hardware
experiments. Um so there can be cases when uh as soon as we start our
controller the robot is just like uh waving the their arms and robots uh like
wildly. Um there can be a couple of issues. So first of all there can be like uh the calibration of IMU is not
accurate enough. So it doesn't have like an accurate uh sense of where it is and
what it's uh what is it it is angular and uh linear velocity and there can
also uh be uh motor that is not producing enough torque. So for example
if we're doing like very agile behavior like climbing up a very high platform or
jumping off a cliff or something like that. Uh sometimes uh in simulation
although we are able to kind of train a policy for the robot to do that the
motor curve is actually different in real and in reality uh on the hardware experiments the motor on the robot will
not be able to produce as much torque as needed for doing such an agile behavior.
So that might also cause some failure on hardware. So
since motors typically have like a rated torque and peak torque, is that something you can cap uh accurately in
simulation or is it pretty hard to set those constraints? Yeah, I think it's pretty hard. So if
we're just using the default parameters from the robot seller, uh I would say uh
it's often not as accurate. So if one wants to do a very accurate calibration
uh I think one has to run a lot of uh experiments with the uh current and
torque and this and speed of the motor to record its behavior. Uh and also
there's another caveat which is when we do more experiments the robot itself
essentially wears off. So that uh relationship between motor current, speed and torque will actually change
over time which is even harder to model in simulation. So I think the best we
can do is try to add more safety guard in simulation. Uh try to penalize a
little bit more uh of the torque uh limits in simulation so that when it transfer to real it won't hit its actual
limit. I see. So in theory, let's say someone were to
be like very conservative and say, you know, they assume the peak torque is
like 50% of the manufacturers's rated torque. Do you think they would still
face any of these challenges or would that be pretty safe? I think in that case um it's more
promising to uh transfer to real without hitting any hardware issue. But uh that
of course limits the range of motion the robot can do. Like if we wanted to do the wolf flip or jumping uh very high, I
think it will be more challenging. Yeah, it's definitely a good point because Yeah, you're pretty much constraining over constraining your
Deploying RL on custom robot
robot. Yeah. Mhm. Exactly. How about So, so far I know a lot of
your work has been on uh deploying it on the Uni Tree robots. So if someone were to try to take their techniques and
deploy on their own custom robot, what do you think would be, you know, some techniques or strategies they would have
to use or understand to do something like that? Yeah, that's a great question. So uh
starting from uh the fundamental level one has to do a very careful CID to
measure the uh for example the inertia and mass of the entire robot as well as
the motors uh and do a careful calibration of all the sensors including IMU encoders and if one has depth camera
uh the cameras as well and then one has to have like a very robust low-level
controller which translates ates the higher level RA policy into lower level
higher frequency torque control and that has to be very reliable in terms of both
the magnitude as well as the frequency the uh command is sent to the torque
sends then to the motor and then uh one has to build like a reasonable enough uh simulation mod in
simulation so that one can start like a arrow training session with a good model
so that lighter when people want to transfer it to real there's a smaller centrial gap.
So for the CIS ID that you talked about are there generally common common
methods or open-source methods that people like to leverage or do they try to make their own?
Uh I think it can be case dependent because uh for the mass you can just take a scale or something like that to
uh measure the mass. Uh I think the most important piece of the CIS ID is
actually amateur or which is the motors or rotational inertia. Uh that is
actually very important. Uh we find out that one of the most important pieces for our SIM to real pipeline. uh and for
others I think there are standardized uh procedure online one can resort to but
uh I think for each robot they have their own uh advantage and disadvantages
and one should be careful about characterizing uh uh if it's the motor
curve that is really important or it's the inertia that is uh causing a lot of
trouble uh for the centuro deployment. So in terms of the accuracy of your
calibration, um how accurate do you think one has to be to have good enough results when they
deploy their RL models? Uh yeah, I think it's again uh case by
case. I would have to say as accurate as one would get so that when during the
training uh phase when doing domain randomization you can randomize uh only in a very small range. Um and uh a very
important thing is that uh maybe during your calibration or CID you only you do
uh multiple rounds and take the average uh and also know the standard deviation of your value so that you can use the
mean and the standard deviation to characterize the range for your domain randomization.
Uh you mentioned uh frequency earlier. Can you dive into a little bit more details about what you were saying for
the frequency part? Uh sure. So the higher level R policy is
running at 50 HzT and the lower level uh torque command is running at 500 hertz.
So the the robot itself has to consume like a relatively higher [snorts]
frequency of torque command while on the higher level the R training is spitting
out like 50 50 Hz uh PD target um for the motor to for the SDK to be converted
into the motor commands. So how how is that typically determined like these
values 50 500 is that something uh that was figured out through experiment or
was it through theory? How how did you guys come to this conclusion?
Uh I think it's uh most of the time u by convention. So for example, unitry SDK
would uh support something like 500 uh uh 500 Hz work command and for the RL
policy I think it's uh mostly trial and error and uh what other people have been
using and have been work uh have been working reasonably well for them. Uh we
will first try the value that is already testified to be working stably. So what
what behavior would you typically see if you went a little bit too high? Say your
RL was running at maybe 100 or even like 200. What sort of behavior might one see
if they were running too high? And what if they went too low like maybe say like
10 hertz? [snorts] Mhm. Uh so I think there's a trade-off
between the compute time and uh the RL like policy frequency. So the higher is
probably better in some sense that we can react to the uh environment more
quickly. Um but there is of course if the policy is running at a higher
frequency then it consumes more uh memory and power on the GPU and
sometimes there will if it's we're running too high we will run into latency problem that the the policy
command cannot actually be sent in real time to the lower level uh motor
command. uh if we're running it on a very low frequency then there's the
problem that we're not reacting to the environment fast enough. If the robot is falling down and it cannot send the
policy command fast enough it might not be able to recover uh immediately. How about in terms of um so you
mentioned a lot about like reaction time. How about in terms of like the model or robot uh stability? Have you
seen any trends in the stability whether it's higher or lower uh frequency?
Uh yeah that's a great question. Uh our intuition is that the lower the frequency the more stable the training
is because we are actually uh exploring on a uh smoother manifold while versus
if it's a higher frequency uh one might be like sampling around a smooth curve
uh in a noisy way. So the command is actually like pretty uh zigzag and
noisy. So that is a an actually like a harder exploration problem in some sense
to actually stabilize your robot. So I think finding the balance between the
smoothness and reactivity is what we uh like what drives us here for 50 Hz.
I see. How about in terms of so like at 50 Hz, you know, your desired points are
kind of spaced apart. So in practice, do you think it's good enough to send those
points at 50 Hz or do you need something that's in between your desired commands
that goes to your low level that does any interpolation? Do you think is there any interpolation that needs to happen
or is that happening in the lower level controllers? Yeah, so that interpolation is actually
on the lower level. So we're essentially uh setting PD position target for the
higher level RL policy which is actually uh implicitly doing torque control using
this PD target uh that we interpolate and translate the position command into
torque command using the PD relationship. So their lower level controller would
you say is like a position P. So you're taking a input as position and then the
position P converts it to torque. Do they have like a cascaded type
controller where there's a position velocity current or is it just position
current? Is that some what's your understanding of their current setup?
[snorts] Um so it's so the for example the unitry SDK is kind of like a black spark
black box to us. So our best understanding is that they are trying to
use the PD relationship to translate the position uh target into torque command.
But sometimes there are weird behavior on the robot. So that might be something
that uh in the blackbox is not per uh performing as we expected. So that that
might also create some gap. So most of the experiments you have done was in uh
position control. Is that right? Have you guys played with torque control directly from your model?
Uh not really. The reason is torque control is uh much less forgiving and it
it should be sent at a very high frequency. So if there is any like non
like any imperfections uh in the command it will be actually amplified by torque
uh torque command versus if it's just PD target you we do an interpolation at a
um at a slower frequency that will be that will not be uh amplified as much as
the higher frequency torque control. So, I'm just curious because like when you have a robot arm that you're tuning, if
it's like let's say if your arm is in the vertical position and it's like
swinging back and forth versus if the arm is like horizontal and swing up and
down, the range of the robot it's in would be very the gains the gains that
were tuned for the different position would be very different because of the load it's seeing. So if you're
controlling it in position and um I'm assuming you have the same gains for a
different position, how does your robot usually still behave with similar
response in the different positions? Yeah, that's a great question. So we are
indeed using the same gain but uh essentially like very small PD gains for
uh for for all the motors. Um that essentially uh like decreases the sim to
real gap because it's like a more gentle uh response to our command send. uh and
I think for of our experiments even including the wall flip and more agile
behavior these gains work perfectly fine for all the motions. So do you think the RL model somehow can
help compensate some of these differences is do you think that's
what's happening because or what's what's your thought on what's happening?
Um I think both the hardware is getting better that there if we have like a
small gain that they can uh uh they can do reasonably well of sending the right
command as well as we during the our training we're like randomly pushing the robots uh for domain randomization. So
uh so even if uh the gains are like uh not the are not reflecting the actual
torque on hardware when we're do randomly pushing the robots the act the robot
actually in simulation the robot actually experiences that a little bit that variation in sim as well. So it has
been trained to see like sort of different motor uh perturbations uh
during simulation. So in your setup when you're doing the simulations um what what specific tool
sets or framework were you using to do your simulation? Uh yeah so we are using Isaac lab as the
training framework uh which is running on issim for the uh lower level
simulation engine. Is there a reason why you guys went with Isaac sim Isaac lab or was it like a choice that the team
RL Simulators (Isaac sim, mujoco, etc)
has made already? Uh yeah so I think it's both because the
Isac lab is highly paralyzable and it has support for distributed training and
so on and it has very good rendering. So later if we want to move on to vision
based RL or uh locomotion uh whole body control it will have uh relatively good
support for rendering for vision as well. How about how about with uh Mujoko? Is
that something you have played with or um any thoughts on that as a simulator?
Uh yeah, so Majoko is uh higher infidelity for the simulation models. Uh
but I think it's relatively uh slower and it doesn't has uh have as good of
the rendering uh support as Isaac but uh I want to note that we're using
Majoko as uh simtosim validation. Oh uh which is saying that
Yeah. Oh, so which is uh saying that we are training the RL policies in Isac and
before deploying it directly on the real robot, we actually run that policy in
Madokco to test if it the dynamics per like the policy with the higher fidelity
dynamics perform well enough in Madoko and if that's robust enough in Madr then
deploy it onto the real hardware. So uh can you go in detail about what
you mean by higher fidelity? Is it like better physics calculation or what what do you mean by that?
Yeah, so major supposedly have it has higher like has a more accurate contact
modeling. So um there are different uh kinds of simulators which are have different trade-off like um mostly the
trade-off between uh simulation accuracy versus the computation speed. So I would
say Isaac is on the higher uh throughput uh end of the spectrum while Majoku is
sort of in the middle where it h has high enough fidelity uh and a reasonably
a reasonable parallelization uh framework. Yes. So uh on the other end end of the
spectrum I would say Drake is probably one of the most accurate uh simulator
where uh it is actually solving optimization problems at each time step to simulate the contact dynamics but uh
it's not GPU uh supported and uh it's hard to parallelize. So there is a a
spectrum of different simulators which have different trade-offs um between computation time and simulation
fidelity. So when you go from Isaac to Mujoko for example,
um have you had specific experiences where when you did do that simtosim
validation where you were able to go back and update your model somehow based
on how it performed? Um yeah so I would say uh because we are
doing so ID uh like well enough and we randomize uh reasonably in uh training
in Isaac SIM the dynamics gap between Isaac and Majorco in our current
pipeline is not that huge. Okay. But the SIM to SIM pipeline also helps
uh us debug debug some other problems. So for example, what should be the
camera latency if we're adding the vision um into the loop? Since we're
doing CIS ID carefully enough and for uh locom motion there is not too much a gap
of dynamics for uh between Isaac and Jooko. So most of the time the dynamics
gap is not as big when we do the simtosim validation but rather some edge
cases that we were not able to detect in Isaac. So for example like what if the
vision latency uh in major in Majoko is uh can be more realistic than I is so
that uh we can detect oh what is the right version latency we should add in
the training process to actually compensate for this discrepancy. So you're saying like Muchoko has uh
longer latency. Is that what you mean by more accurate? Um so uh in training when we render the
vision in Isac the simulation will actually pause to uh let the simulation
uh simulator render the the vision but during the deployment in modroo when
we're actually running the policy it will actually not wait for the simulator to render the image so there will be
some sort of latency in that regard. Okay. So is it possible to create a fake
latency in Isaac to simulate that behavior or is it pretty hard to
Yeah. So what we do is that uh we create a buffer of the sensor uh readings and
it we we do kind of a CIS ID Unreal to see what the latency is for like each
sensor and then we chose the the reading in the buffer which is around that time
range. H. So if you're able to do that then technically would you still have to
do this muchoko verification if you're able to create a very realistic latency
in Isaac or do you think that step would still be necessary? Uh yeah so I think uh another advantage
of lat uh of major in addition to like uh verify that latency is uh running the
deployment code in simulation first. So I it's actually might be kind of uh
complicated to uh run the deployment code in isac directly but uh in major
code it's a much more direct interface for us to uh deploy our uh inference
code. So in addition to testing the vision discrepancy dynamics gap there
will there's also the layer of we are testing our deployment code in
simulation. What what's the main challenge with um running inference in Isaac?
I think there is just a lot of abstraction layer in Isaac. Um, and it's
less direct of an interface than Majoko for us to uh like uh deploy our uh
inference code because in Mujoko they let you like send direct position or torque commands just
based on you know how you set up your XML file, right? So it's like a pretty
straightforward way to command it. Right. That's kind of what you mean.
Mhm. Okay. Okay. Cool. Um, so in general, if someone is like trying to learn more
Resources for learning RL
about RL, whether it's like the application side or theory side, what
would you say is a good starting point for someone to kind of get into the topic?
So for the theory side, like Richard Sutton's introduction to reinforcement
learning is uh one of the primer book. uh and for for from for example from
people from a controls perspective a dimitary bersa's uh dynamic programming
and optimal control would be a very like uh control perspective uh way to explain
the RL concept and there are also some like Berkeley courses taught by professor Sergey Levvin uh on uh
reinforcement uh learning and deep learning that one can also like watch it online. So I think for the theory side
there are a lot of resources either it be online video or books one can uh
refer to and for the practical side I think it's reading others codebase for
RL deployment and try to adapt them uh for one's own use case so that one can
get more and more hands-on experience on the training and uh deployment.
Kinematic retargeting
Very nice. Those are some really good useful resources. So um definitely look into those. Um I know we'll probably
spend the second half of this or not second half but this second part talking about some of your specific applications
that you worked on. So you know a lot of the new cutting edge work is probably you know the research that you've done
on like retargeting. So maybe we could start start looking into that topic
right now. And maybe just for those that have never heard of kinematic retargeting, can you give like a highle
overview of exactly what that is and what problem you're trying to solve?
So the kinematic retarding is basically transforming human motions onto robot
motions. Since the humanoid robot looks very much like the human, we want to
reuse the human motions to direct our search for how we command the robot. So
say here's a task of human picking up the box and we want to transfer the same
motions onto the robots picking up the box. So there are some standard ways of doing so such as defining some key
points on the rob on the robot and the human um and try to match the absolute
position between the two. But there are some problems of doing so. Uh for
example, because the humanoid can be much shorter than the human, this direct
uh scaling and translation matching will will result in some penetration.
And here we're using some technique to avoid this issue. Penetration you mean like
uh in going into itself? Is that what you mean? Uh yes. So let me try to show a direct
example. something like this. So the keyoint matching is the technique I was
describing as the standardized technique for the uh humanoid human to humanoid
kinematic targeting pipeline which is essentially choosing some key points on the human and try to match the same set
of semantic key points on the robot to the absolute position of these key points on the human. Um and this because
uh the humanoid can be much shorter than the human directly matching this absolute position can result some
penetration with the object. So for example something like this and this is essentially um a very um
like a very direct result of the different scale of human and the robot
because uh say imagine the human is like say 1.8 8 m while the humanoid we're
using is 1.3 m. Picking up the same box will actually
result in same in different relative skills for the human and the robots. So
directly doing this key point matching will result in some artifacts like penetration.
So uh you talk about going from um like key points from a human. Is the main
idea to get videos of people or what's the main are you trying to utilize like
the whole internet data to do some of this? Like what's the bigger picture idea that um this method would end up
being used for? Yeah, that's a great question. So currently we're using motion capture
data which is essentially a human demonstrator wearing a very specialized
mocap suits in a specialized um room with cameras that can accurately
identify the position of the human. [clears throat] Um but this sort of data is very
expensive and uh ultimately we want to utilize the videos of the entire
internet to cheat teach how teach the robots to do the things but there are some challenges uh in this regard. So uh
the uh re 3D reconstruction of human and objects from video is a very non-triv
trivial research topic. Some of the problems involve like some the human
root will kind of be floating in the air and uh going back and forth. So how to
extract uh robust, reliable and realistic data from the video is a very
um challenging and but also interesting research topic. I see. So I guess you guys are kind of
assuming that the video to model part is
handled, right? And you're just focusing more on if you already have the key points. Is that right?
Uh exactly. Yeah. Okay. So um you were mentioning like oh if the robot is smaller then um you're
trying to focus on having like a bigger human where the key points are bigger to something smaller. Do you think your
current method can also work in reverse? If the human is smaller but the robot is
bigger, can it also handle those cases? Like is it general enough to do that? Absolutely.
Yeah, absolutely. Um so the way our method works is um let me try to share
my screen again. We tried to build something called interaction mesh which
is uh in addition to defining key points on the human we also define key points
on the object. So let me give a more concrete example here is a human picking up the box and
we want to transfer its motion to the robot picking up the box. uh as I
mentioned we select some key points semantically important key points on the human and the set of same key points on
the robot. We also define key points on the object and use the same set of key
points on the uh uh on the object for the robot as well. Then we build uh an
interaction mesh which is a volutric structure that uh captures the relative
uh information uh position information between the human and the object.
So uh here is how the mesh looks like. So as we can see it not only captures
the information between the human joints themselves but also how it relates to
the object. So uh in this example, the human's right
hand uh is touching the right face of the object and we want the robot's right hand to also touch the right surface of
the object that is actually captured by the um
graph structure that preserves the relative spatial information in this motion.
Is there a minimum number of points you need for the object to have the full
understanding of your object? Uh that's a really good question. So uh
ideally we want the contact points between the human and the object. So say
in this point there might be two key points which is like uh the left uh hand
and the right hand uh touching the surface of the box. Um but in order to
make the algorithm more robust uh we actually uh select more key points than that. So uh we essentially randomly
sample like say 20 to 50 points on the object to keep the relationship between
the robot human uh and the object. Okay. So how how well because right now
we see an example with a box. How well do you think your current method could extend to, you know, more deformable or
organic looking objects like, you know, maybe a pillow, uh maybe like a teddy bear, like something or like a blanket
even like how how would this method extend to something like that? Uh I think uh this retargeting method
will be directly applicable to uh all kinds of different objects including
deformables. Um, and because essentially we're capturing the relationship between
the human and some points on the object. As long as we can define the key points,
either it be contact points or it be semantically meaningful key points, uh, our retargeting pipeline can be directly
transferred. Okay, very cool. Um so I know a lot of like the general
Generalization in robotics (generalist vs specialist)
trend that we're seeing in robotics is um you know there's still a lot of companies where they have very special
models that do very specific tasks but then there's also companies like Tesla
and some other companies that's trying to do of a more of a endtoend model where you know they only get input is
the video that they see of the world and the output is the motor actions like
let's say a very high level task. Go clean the room or you know go get me coffee, right? Something that high of a
level of a task. Do you feel like in the future is that the direction that
robotics is headed where there's such a general model that can do anything or do
you feel like we still need very specialized models that does very specialized tasks? So uh this is a great
question and a very hot topic that uh both industry and academia has been debating a lot. So I personally think
would lean more towards a generalist policy. Uh the reason is that um
multiple skills that are trained for the generalist policy can might be able to
um transfer and help each other generalize. Um and as sometimes uh the
generalist policies uh the advantage people believe is that it can learn something like a common sense or the
intuition intu uh or intuitive physics which is uh roughly like a model of how
the world will react that can actually transfer across different task. So if
once the model uh gains this common sense, it will be able to more easily
transfer to a new task it has never seen it before. So, do you think these models
eventually can get down to like millimeter precisions like for example
if um like very hard tasks for example
like surgery maybe or even like if they're trying to assemble PCB boards
onto or put you know IC parts onto a PCB board. Do you think these models can
eventually get to that level of precision or do you think they probably
still need, you know, very like more of the typical robot programming that we
think of where you program exact positions. What what's your thought on that?
Uh yeah, that's a great question. I think it might be hard for these generalist policy to directly perform
millimeter accuracy task uh directly out of the box. But if we just collect a
small number of uh demonstrations on these specific tasks and do post
training that is to refine the pol generalist policy on our specific task
with the in-domain data and specific data I think we will be able to achieve
very high accuracy. Uh the old traditional like classical methods like scripting the uh scripting uh the robot
arms can achieve very high fidelity but they are more brittle to uh longtail uh
problems and they might do less well in terms of like vision and uh the more
semantic reasoning where these generalist policy might be able to learn from failures or recovery from the other
skills and try uh try to directly recover from something uh that uh it
hasn't been seen in the classical scripting kind of method. So I know you mentioned like having more
Data and training for AI models, data augmentation
data for those specific cases. Um in general, do you feel like what's
stopping us from having robots in our house that's working? Do you feel like that's more of a data problem or is it
more of a model and architecture or even
robot hand development problem? What's your current take on that? Yeah, I think um the current obstacles
are multiffold. uh I would say the humanoid hardware is uh relatively more
robust than say the hand hardware. So the sim to real gap is smaller and the
motor control is more precise. Uh but of course uh in addition to the hardware
problem there is also the software problem and uh for a lot of researchers
the software the core of the software problem is the data problem. Um so I
think some of our hypothesis is that uh if we have uh enough high quality data
maybe the training architecture and policy architecture doesn't matter as much.
So we are essentially trying to control the quality of the policy output uh by
controlling the uh data quality directly. So if we can get uh very high
quality um data for the robots uh I think it will be a very important uh
improvement for reliable deployment of these robots. So right now a lot of people are you
know either manually manually getting data or using data from like
simulations. Um I know Nvidia recently have has been pushing Cosmos which is
their uh AI data. Basically they're generating video data synthetically and
they could augment and do like data transfer for different scenes. For
example, if it's cloudy, sunny, they could augment all of that. um do you think that is the right approach to
getting more data or do you think there's uh different ways someone should be focusing on to get more data?
Yeah, that's a great question. So I think we should actually leverage data
from all different kind of resources. uh either it be the most expensive but the
uh arguably the highest quality data which is the teleoperation data on the
real robot or it be the simulation data which we can generate in large scale but
always has the sim to real gap. Uh and the also there are also kinds of the
other uh data which is for example internet video data uh world model data
uh that has rich uh semantic and visual features for like the uh video uh models
but might has less action data. So I think different data has their own
advantages uh and their own uh specialized um targeting area. Um and
combining these different sources of data together to enable both control and
dynamics accuracy as well as semantic and uh visual understanding is a very uh
I think a very interesting and promising topic. So I guess if you were to put a
allocation like if I imagine like if there's a pie chart and you were to allocate like the percent of each
category that you mentioned just roughly you know roughly speaking how would you
categorize each of the parts of the pie for the different types of data.
Yeah, since myself is working on data generation in sim and doing kinematic retargeting, uh my answer will obviously
be skewer towards using simulation data. Uh it's uh very uh scalable and it can
give us reasonably accurate uh dynamics and action uh data. uh while so I would
say I would allocate like half of the effort in uh doing uh in generating
simulation data and the other half is split between uh video and uh real world
teleoperation data. So I think video data is also more
scalable than the real world teleoperation because for the tele operation you always need a human
operator to operate the robots. uh it can be time consuming uh it kind of
cause fatigue for the human operator uh and wears the robot hardware right
uh for the video you can we have the entire internet and we have these like
you mentioned cosmos the generative uh video models word models that can
generate like essentially endless video data for us to capture the uh visual
dynamics uh semantic features and so on so I think that one is more scalable. So
I think my personal take is to uh spend as much effort uh as possible on the
scalable uh like methods including simulation and video and also allocate a
reasonable amount on the real data to actually close the sim to real gap. Uh there's been a lot of topic on you
Traditional optimization vs RL
know traditional and newer ways of doing RL. Can you kind of dive into some of the details of that and maybe the
differences between the two? Uh yeah. So um I think I want to give
the kinomide retargeting as an example of doing um things in uh both the
classical optimization based and model based uh perspective and the more learning based perspective. So my very
first background is actually in optimization and modelbased control and
that actually lays a foundation for me to write omni retarget which is an
constrained uh optimization based kinematic retargeting pipeline and this
sort of optimization based uh pipeline will enable something that is not quite
achievable by a learning based pipeline. So because we're reasoning about hard
constraints kinematics in a optimization way uh fashion we can enforce higher
quality. So we can actually enforce hard constraints that learning based policy won't be able to enforced. So like say
we don't want penetration of of the object. We don't want the joint to
exceed its hard limits. We don't want the velocity to ex exceed a certain
threshold. For us, we can write it as hard constraints in the optimization program versus uh in the more learning
based method, people normally put it as soft penalty in the cost or reward and
then try to optimize it. Uh it's not sometimes it's not guaranteed that these
hard constraints are actually enforced. So there might be a little bit of penetration or joy limit violation if
we're doing this kind of soft penalty. I see. But uh by doing the hard uh constraint
uh optimization based uh pipeline, we're able to enforce these hard constraints very systematically and rigorously so
that we can have higher quality data to then be consumed by downstream learning
paradigms. So do you think it's possible to take both traditional optimization
and RLbased methods together or do you think it's more of a eitheror type of
situation? So uh yeah so the combination of both is
actually my goal. So upstream I'm using this optimization based hard hard
constraint uh formulations to generate hard quality data and downstream where
training our policies to track this high quality data. So the combination of very
rigorous uh high quality data generation plus the massively parallelizable RL
training I think is a very promising paradigm. H. So you're saying the the
main hard constraint is more like the highlevel loop closure in a way
that's making sure the robot doesn't break these constraints. Is that how you would describe it? Or how for like the
general audience, how would you kind of describe how that's able to keep
everything, you know, under the main constraints that you want it?
Um yeah, I would say uh if we want to enforce like uh important hard
constraints uh and specifically for the army project in a kinematic level
uh which is just the robots abain its uh morphological constraints we can do it
in a systematic optimization based way and later when we translate into
physically uh plausible dynamically plausible uh behavior we want to
leverage the uh large scale uh parallelization in simulation in Isac to
do this translation. So it so like on a higher level if we can split the
problems into two phases where like uh smaller number of computation but but
requires higher quality uh we can do it with the modelbased strategy but uh for
the uh lower fidelity requirement and uh and like massively parallelizable uh
setup we can use the learning paradigm. But when you're trying to combine the two, um I guess if you were to try to
describe the architecture of your program, like how would you lay out the
pieces? So like for example, I'll just give you like a simple example of what I mean by that. Like when you have uh like
a RL model for example of a robot walking, um the blocks I would describe
might be like on the left I have like my RL. Okay, like on the very left I might
have a block that's like my desired trajectory and the input of that goes to some RO inference model that's computing
the desired torque and then to the right of that might be feeding it to the torqus of the actuator. So if you were
to kind of describe in like a block diagram level structure of a hybrid
approach where you have your um traditional optimization technique and
your RL uh techniques, how would you kind of describe that
visual picture of how data is flowing? Yeah. So I would say um the model based
I I'm I'm personally using the modelbased approaches to generate higher quality data and then uh it it will be
used as essentially as initial guess for the RL policy. So uh we can train our
policy from scratch but that is very time consuming requires a lot of reward
tuning uh and uh like the uh the the behavior resulting behavior might be uh
less natural for example but with the help of modelbased methods we will be
able to enable um more fluid u motion uh
from the uh initial guess provided by the modelbased methods. And they then
the arrow policy like just bootstraps or initialize from that to learn uh a
better um controller. Oh, so you were talking about hard constraints using um like traditional
optimization based techniques and also combining that with RL techniques. So
can you kind of walk a little bit into more detail about how you take the two things and combine them together?
Uh yeah sure. So um as I mentioned uh before we do the kinematic retargeting
by building a graph that preserves the relative location between the human and
the robot uh as well as the objects. So here is the optimization program I'm
solving. I'm trying to there are uh components that are the objective as
well as the constraints. So the objectives are encouraging this interaction to be preserved. Uh and the
hard constraints including non-penetration where we don't want the robot hand to
penetrate the object to go into the object. So we want to penetrate that. We want the joint to stay within the limits
and as well as the velocity to stay within the speed limit. And we also want a hard constraint that the food doesn't
skate while the robot is walking. Otherwise the robot will just be sliding all the time. So these constraints
together with the interaction preserving objective uh will give us some very high
quality data that uh preserves the human motion of picking up a box onto the
robot uh as well as so uh as satisfying all the hard constraints including the
robot hand not penetrating the object and the food is not sliding while walking. So would you say you're using
this um constraint here as the input to your RL or are you using this constraint
to generate like a full series of like motion data like how how would you say
it's the right way to understand? Uh yeah so this is actually kind of a
hierarchical framework. So f first we use this pipeline to generate data with
hard constraints so that it can be used as initialization for the RL. So during
RL training, we actually initialize the uh the agents to be in some random
position or uh in some configurations at random time step in this uh in this
motion data set and then from that the RO be will be able to start with from
these configurations and bootstrap from that to come up with a dynamically feasible solution. So uh say let's
compare it with a training from scratch paradigm where the uh the arrow policy
initialized the agent to be in random modocation. In that kind of scenario there can be
there can be all kinds of penetrations uh joint limits violation velocity limit
violation as well as food skating in that kind of uh initialization from
scratch. So uh with this optimization based uh qual uh high quality data
generated the RL will be initialized from a much better configuration than
just initializing from scratch. So how how do you know that the initial
position if the initial position is in something that's like physically
possible? How do you know the rest of the RL execution will also be physically
possible? What is kind of enforcing that? Yeah, so that's actually just
timestamping the simulator in Isac that is enforcing the dynamical constraints.
So it's still like checking your optimization equation every time. Is that what you're saying? like your RL.
Oh, so uh the uh data that comes from my
optimization is simply used as the initialization for RL and then RL just
does whatever uh it's supposed to do in ISC. Okay. So as long as you're saying as
long as the initialization when you say initialization I guess you mean like the initial position it's in for like each
episode. Is that the correct way to describe it? Uh yes yes uh not only does
the reference motion as uh a act as a good initialization for the RL policy
but it also adds a guidance as a guidance for the RL policy. So uh it u
tells the uh RL policy that where the robot should go at the next time step
and the RL tries to achieve that with the current dynamical constraints in
Isaxim. All right. So, that's it for this episode. Uh, thank you Lou for coming on to this podcast show and I'll leave some
links in the video description for some of her works so you guys can go ahead and check that out.
