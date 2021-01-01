---
title: The ethics of AI
subtitle: >-
 How to create responsible system with right values?
date: '2020-12-10'
author: src/data/team/jayesh-ahire.yaml
categories:
    - src/data/categories/general.yaml
    - src/data/categories/news.yaml
tags:
    - Artificial Intelligence
    - Ethics
    - Machine Learning
    - Tech talks
excerpt: >-
 Nowadays AI is permeating an ever wider range of our lives. It is associated with great hopes, but it also raises fears. Therefore, the call for ethical guidelines regarding the new technologies is becoming increasingly louder.
thumb_image: images/classic/post-5.png
image: images/classic/post-5.png
image_position: right
template: post
---
 
Whether in daily mobility, in industrial applications or in the form of assistance solutions at home: Artificial Intelligence permeates an ever wider range of our lives. It is associated with great hopes, but it also raises fears. Therefore, the call for ethical guidelines regarding the new technologies is becoming increasingly louder.

There are various questions we need to find answers to:
- How can the “right” values be integrated into technical action? 
- Can and may AI determine what is right? 
- How to ensure that AI is developed, used and provided in the service of mankind? 
- And how can the decision-making processes of AI, including wrong decisions, be made transparent, given the huge, complex amounts of data?

We organized a panel discussion on what is importance of implementing ethical practices within your predictive models, data workflows, products and AI research. I was the part of the panel along with [Scott Haines](https://dev.to/newfront), [Lizzie Siegle](https://dev.to/lizziepika) and [Nick Walsh](https://dev.to/thenickwalsh). 

In this article, we will go through some of the points we discussed with panel and their views on various topics along with my view on each topic.

For the sake of simplicity, we will discuss further details into 3 broad categories:
1. Fairness & Bias
2. Interpretability & Explainability
3. Privacy

Let's jump into it!

## 1. Fairness & Bias:
Bias is often identified as one of the major risks associated with artificial intelligence (AI) systems. As A Survey on Bias and Fairness in Machine Learning suggests, there are clear benefits to algorithmic decision-making; unlike people, machines do not become tired or bored, and can take into account orders of magnitude more factors than people can. However, like people, algorithms are vulnerable to biases that render their decisions “unfair”. In the context of decision-making, fairness is the absence of any prejudice or favoritism toward an individual or a group based on their inherent or acquired characteristics. Thus, an unfair algorithm is one whose decisions are skewed toward a particular group of people.

<img src="https://languagelog.ldc.upenn.edu/myl/SMBC_AI_Firing.png" width="400" height="300" />

The public discussion about bias in such scenarios often assigns blame to the algorithm itself. The algorithm, it is said, has made the wrong decision, to the detriment of a particular group. But this claim fails to take into account the human component: People perceive bias through the subjective lens of fairness.

So, Bias occurs when we discriminate against (or promote) a defined group consciously or unconsciously, and it can creep into an AI system as a result of skewed data or an algorithm that does not account for skewed data. Fairness, meanwhile, is a social construct. And in fact, when people judge an algorithm to be “biased,” they are often conflating bias and fairness: They are using a specific definition of fairness to pass judgment on the algorithm. 

No AI system can be universally fair or unbiased as there are more than 20 definitions of fairness and it is impossible for every decision to be fair to all parties. But we can design these systems to meet specific fairness goals, thus mitigating some of the perceived unfairness and creating a more responsible system overall.

This discussion also brings us to a question if we can relate trolley problem here? 

Let's first understand what is trolley problem:
> A runaway trolley is heading down the tracks toward five workers who will all be killed if the trolley proceeds on its present course. Adam is standing next to a large switch that can divert the trolley onto a different track. The only way to save the lives of the five workers is to divert the trolley onto another track that only has one worker on it. If Adam diverts the trolley onto the other track, this one worker will die, but the other five workers will be saved.

![](https://thumbs.gfycat.com/SneakyUnrealisticFlee-size_restricted.gif)

*Should Adam flip the switch, killing the one worker but saving the other five?*
The original trolley problem came from a paper about the ethics of abortion, written by English philosopher Phillipa Foot. It’s the third of three increasingly complicated thought experiments she offers to help readers assess whether intentionally harming someone (e.g. choosing to hit them) is morally equivalent to merely allowing harm to occur (choosing not to intervene to stop them getting hit). 

Apart from above version there are various modern day implications of this problem which are considered while social psychology studies conducted by different universities. You can try [Moral Machine](http://moralmachine.mit.edu/) by MIT. 

Who else is ready to accept this as a solution?
![](https://thumbs.gfycat.com/HomelyIckyBlackfly-size_restricted.gif)

Jokes apart, But is this applicable in case of AI?
[Simon Beard](https://qz.com/author/simon-beard/) explains in his article that this philosophical issue is irrelevant to self-driving cars because they don’t have intentions. Those who think intentions are morally significant tend to hold strong views about our freedom of will. But machines don’t have free will (at least not yet). Thus, as far back as Isaac Asimov’s three laws of robotics, we’ve recognized that for machines to harm someone—or “through inaction, allow a human being to come to harm”—is morally equivalent.

Which seems logical explanation as AI because trolley problem is very situational and AI works in real-world scenarios which are extremely uncertain and you can't certainly apply universal laws of ideal behavior to machines. AI tends to learn continuously from it's mistake and try to achieve a state with better decision making capabilities with each iteration.

This brings us to second point of our discussion that how important explainability and Interpretability of decisions made by AI is considering ever increasing lack of transparency has only become more visible, midst of techlash.

## 2. Interpretability & Explainability
When it comes to machine learning and artificial intelligence, explainability and interpretability are often used interchangeably.

![](https://i.imgflip.com/2yc1nf.jpg)

According to [Miller](https://arxiv.org/abs/1706.07269), interpretability is the degree to which a human can understand the cause of a decision. Interpretable predictions lead to better trust and provide insight into how the model may be improved. We can also define intepretability as the extent to which we are able to predict what is going to happen, given a change in input or algorithmic parameters. 

Explainability, meanwhile, is the extent to which the internal mechanics of a machine or deep learning system can be explained in human terms. It’s easy to miss the subtle difference with interpretability, but consider it like this: interpretability is about being able to discern the mechanics without necessarily knowing why. Explainability is being able to quite literally explain what is happening.

Let's understand using example, consider a chemistry experiment: Knowing that when you mix 1 gm. of blue colour chemical with 100 ml of liquid you get pink colour solution and this is interpretability but understanding exact chemical reaction for this experiment means explanability.

Interpretability & Explanability are not required if the model has no significant impact or when the problem is well studied. This might also enable people or programs to manipulate the system but if we’re unable to properly deliver improved interpretability, and ultimately explainability, in our algorithms, we’ll seriously be limiting the potential impact of artificial intelligence. Which would be a shame.

![](https://media.wired.com/photos/5c871a4592cce301f3dfae3a/191:100/w_955,h_500,c_limit/TopArt_Final_04.gif)

Let's discuss few methods to increase the interpretability of complex ML models.

- **LIME or Local Interpretable Model-Agnostic Explanations**, is a method developed in the paper [Why should I trust you?](https://arxiv.org/abs/1602.04938) for interpreting individual model predictions based on locally approximating the model around a given prediction. The researchers explain that LIME can explain “the predictions of any classifier in an interpretable and faithful manner, by learning an interpretable model locally around the prediction.”
What this means in practice is that the LIME model develops an approximation of the model by testing it out to see what happens when certain aspects within the model are changed. Essentially it’s about trying to recreate the output from the same input through a process of experimentation.

- **DeepLIFT (Deep Learning Important FeaTures)** is another method which serves as a recursive prediction explanation method for deep learning.  This method decomposes the output prediction of a neural network on a specific input by backpropagating the contributions of all neurons in the network to every feature of the input. DeepLIFT assigns contribution scores based on the difference between activation of each neuron and its ‘reference activation’. DeepLIFT can also reveal dependencies which are missed by other approaches by optionally giving separate consideration to positive and negative contributions.

- **Layerwise Relevance Propagation (LRP)** is a technique for determining which features in a particular input vector contribute most strongly to a neural network’s output. The technique was originally described in [this paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140). It defines a set of constraints to derive a number of different relevance propagation functions.

- **Algorithmic generalization:** Improving generalization is not as easy as it sounds. Sometimes Models feel secondary when we realise that most ML engineering is applying algorithms in a very specific way to uncover a certain desired outcome. However, by shifting this attitude to consider the overall health of the algorithm, and the data on which it is running, you can begin to set a solid foundation for improved interpretability.

- **Pay attention to feature importance:** Feature importance tries to answer question, what features have the biggest impact on predictions. Looking closely at the way the various features of your algorithm have been set is a practical way to actually engage with a diverse range of questions, from business alignment to ethics. Debate and discussion over how each feature should be set might be a little time-consuming, but having that tacit awareness that different features have been set in a certain way is nevertheless an important step in moving towards interpretability and explainability.

So, we explored the views around to open a black-box or not! Now, we have to think about privacy aspect which is one of the major concern in today's world.

![](https://media.giphy.com/media/lea5DJxF4Lz4k/giphy.gif)

## 3. Privacy
When we look around for application areas of AI, we find that much of the most privacy-sensitive data analysis today–such as search algorithms, recommendation engines, and adtech networks–are driven by machine learning and decisions by algorithms. As artificial intelligence evolves, it magnifies the ability to use personal information in ways that can intrude on privacy interests by raising analysis of personal information to new levels of power and speed.

The discussion of AI in the context of the privacy debate often brings up the limitations and failures of AI systems. Bias in data and unfair systems have raised significant issues, but privacy legislation is complicated enough even without packing in all the social and political issues that can arise from uses of information. As Cameron F. Kerry suggested in one of his survey, to evaluate the effect of AI on privacy, it is necessary to distinguish between data issues that are endemic to all AI, like the incidence of false positives and negatives or overfitting to patterns, and those that are specific to use of personal information.

The privacy legislative proposals that involve these issues refer to decisions made by AI as “automated decisions” (borrowed from EU data protection law) or “algorithmic decisions”. In general I believe that this helps to shift people’s focus from the use of AI as a tool to the use of personal data in AI and to the impact this use may have on individuals. This debate centers in particular on algorithmic bias and the potential for algorithms to produce unlawful or undesired discrimination in the decisions to which the algorithms relate. These are major concerns for civil rights and consumer organizations that represent populations that suffer undue discrimination.

There's lot to think about and discuss about the scope of privacy legislation as some of them also address algorithmic discrimination. 
Let's split our views into two separate points:
1. to what extent can or should legislation address issues of algorithmic bias? According to Law, Discrimination is not self-evidently a privacy issue, since it presents broad social issues that persist even without the collection and use of personal information, and fall under various civil rights laws.
2. When we want to ensure privacy protection in AI systems and regulate use of consumer data and try to stop this unfair and deceptive practices the consumer choice based notice-and-choice model for privacy policies become meaningless. We will require a change in the paradigm of privacy, to protect such privacy interests in the context of AI. 

we need an approach that addresses risk more obliquely, with accountability measures designed to identify discrimination in the processing of personal data and regulate use of it. Number of organizations and companies as well as several legislators propose such accountability. Their proposals take various forms:

- *Transparency:* Accountability is a major concern and being able to examine how company is handling data makes it much more easier to hold company accountable for things. This mostly refers to disclosures relating to uses of algorithmic decision-making. Replacing current lengthy, detailed privacy policies are not helpful to most consumers with “privacy disclosures” that require a complete description of what and how data is collected, used, and protected would enhance the benchmarking which is being done by regulators. In turn, requiring that these disclosures identify significant uses of personal information for algorithmic decisions would help watchdogs and consumers know where to look out for untoward outcomes.

- *Explainability:* As we discussed above, this helps us understand how the data has been used to reach to particular decision and which features have played significant role in conclusion. European Union’s General Data Protection Regulation (GDPR) also follows this approach to settle disputes because of automated decisions. The GDPR requires that, for any automated decision with “legal effects or similarly significant effects” such as employment, credit, or insurance coverage, the person affected has recourse to a human who can review the decision and explain its logic. This incorporates a “human-in-the-loop” component and an element of due process that provide a check on anomalous or unfair outcomes. The central problem with both explainability is that you’re adding an additional step in the development process along with significant regulatory burden. Indeed, you’re probably adding multiple steps. From one perspective, this looks like you’re trying to tackle complexity with even greater complexity.

- *Risk assessment:* When we talk about risk assessment in this context we are mostly dealing will assessing impact of algorithmic decision on individuals and potential bias in design of a system. As Cameron F. Kerry rightly points out, for the regulatory burden to be proportionate, the level of risk assessment should be appropriate to the significance of the decision-making in question, which depends on the consequences of the decisions, the number of people and volume of data potentially affected, and the novelty and complexity of algorithmic processing.

- *Audits:* There are some general accountability requirements as well as ways such as self-audits or third-party audits which help authorities to ensure companies comply with their privacy programs. When we couple auditing outcomes of AI decisions with proactive risk assessments it can help us match foresight with hindsight; although, like explainability, auditing machine-learning routines is difficult and still developing.

Because of the difficulties of foreseeing machine learning outcomes as well as reverse-engineering algorithmic decisions, no single measure can be completely effective in avoiding perverse effects. Thus, where algorithmic decisions are consequential, it makes sense to combine measures to work together. That's why it’s important for algorithm operators and developers to ensure that they won't leave some groups of people worse off as a result of the algorithm’s design or its unintended consequences. [[AI debate](https://www.brookings.edu/research/algorithmic-bias-detection-and-mitigation-best-practices-and-policies-to-reduce-consumer-harms/), Nicol Turner Lee with Paul Resnick and Genie Barton]

![](https://www.washingtonpost.com/sf/national/wp-content/uploads/sites/11/2017/12/o-jibo1210.gif)

### On a positive note,
Amid the dissonance of concern over unfair decisions made by artificial intelligence (AI) and use of personal data. The potential for AI to do good cannot be overlooked. Technology leaders such as Microsoft, IBM, Google and many others have entire sections of their business focused on the topic and dedicate resources to build AI solutions for good and to support developers who do. In the fight to solve extraordinarily difficult challenges such as Accessibility, Climate Change, Conservation and the Environment, World hunger, Human rights, Fake News and so many others, we can surely use help by AI!

`Note`: In case you're interested to watch this discussion, you can check out the video here:

{% youtube pPwJqIAZ4QE %}

### References:
1. [Importance of Interpretability](https://christophm.github.io/interpretable-ml-book/interpretability-importance.html)
2. [MIT's self-driving cars survey for Moral Machines](https://www.technologyreview.com/2018/10/24/139313/a-global-ethics-study-aims-to-help-ai-solve-the-self-driving-trolley-problem/)
3. [can we open the Black Box?](https://www.scientificamerican.com/article/can-we-open-the-black-box-of-ai/)
4. [What do we do about the biases in AI?](https://hbr.org/2019/10/what-do-we-do-about-the-biases-in-ai)
5. [what is fair?](https://www.strategy-business.com/article/What-is-fair-when-it-comes-to-AI-bias?gko=827c0)
6. [A Survey on Bias and Fairness in Machine Learning](https://arxiv.org/pdf/1908.09635.pdf)
7. [Explainability & interpretability in AI](https://www.kdnuggets.com/2018/12/machine-learning-explainability-interpretability-ai.html)
8. [protecting privacy in an AI driven world](https://www.brookings.edu/research/protecting-privacy-in-an-ai-driven-world/)
9. [The trolley dilemma in AI](https://strongbytes.ai/the-trolley-dilemma-in-ai/)
10. [The problem with the trolley problem](https://qz.com/1716107/the-problem-with-the-trolley-problem/)
11. [Layerwise Relevance Propagation](http://danshiebler.com/2017-04-16-deep-taylor-lrp/)
12. [AI & Privacy](https://privacyinternational.org/learn/artificial-intelligence)

 
 
| About author                                                                                                                                                                                                                                                                          |                                                                                                               |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------  |-------------------------------------------------------------------------------------------------------------  |
| Jayesh is a founding engineer at Traceable and He loves reading, writing poems, talking tech, philosophy and you can find him on [twitter](https://twitter.com/jayesh_ahire1) and [linkedin](https://linkedin.com/in/jayesh-ahire) to discuss around anything.   | ![](https://avatars0.githubusercontent.com/u/26570044?s=460&u=b16d66186bbbcf75374de557e36c9174fbe68272&v=4)   |
