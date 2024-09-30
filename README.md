# Deep Learning & Machine Learning Resources
This is a comprehensive list of machine learning and deep learning resources. Resources cover the theory and implementation of LLM and image diffusion models. If you'd like to contribute a resource then please open a new [issue](https://github.com/dgwyer/ml-resources/issues/new) or submit a [PR](https://github.com/dgwyer/ml-resources/pulls).

## Work In Progress
This is a very new list of ML resources so the structure may change around while we figure out the best way to organize the resources. By all means [let me know](https://github.com/dgwyer/ml-resources/issues/new) what you think and how it can be improved!

As you may have noticed most (or almost all!) resources focus on diffusion models. This is because where most of my focus has been these last few months. But I do want to transition to cover LLMs too. So if you have some good language based resources you'd like to share please [let me know](https://github.com/dgwyer/ml-resources/issues/new).

## Table of Contents

- [Deep Learning \& Machine Learning Resources](#deep-learning--machine-learning-resources)
  - [Work In Progress](#work-in-progress)
  - [Table of Contents](#table-of-contents)
  - [Research Papers](#research-papers)
    - [Websites](#websites)
    - [Papers](#papers)
  - [YouTube Channels](#youtube-channels)
  - [Individual videos.](#individual-videos)
    - [Training \& Fine Tuning](#training--fine-tuning)
    - [Transformers](#transformers)
    - [Diffusion](#diffusion)
  - [Podcasts](#podcasts)
  - [Newsletters](#newsletters)
  - [Courses](#courses)
  - [Books](#books)
    - [Python Books](#python-books)
  - [Blogs](#blogs)
  - [Blog Posts](#blog-posts)
  - [GitHub Repositories](#github-repositories)
    - [Student Repos](#student-repos)
    - [Code Snippets and Examples](#code-snippets-and-examples)
  - [Discord Channels](#discord-channels)
  - [Software and Tools](#software-and-tools)
  - [Cloud Services](#cloud-services)
  - [Datasets](#datasets)
  - [Twitter](#twitter)
  - [Interviews](#interviews)
  - [Jupyter Notebooks](#jupyter-notebooks)
  - [Other Resource Lists](#other-resource-lists)
  - [Additional Resources](#additional-resources)

## Research Papers

### Websites

- [arXiv](https://arxiv.org/) - Main site
  - [Artificial Intelligence](https://arxiv.org/list/cs.AI/recent)
  - [Machine Learning](https://arxiv.org/list/stat.ML/recent)

### Papers

A collection of some of the seminal papers on machine learning, and deep learning.

- [A Survey on Generative Diffusion Model](https://arxiv.org/abs/2209.02646) [[PDF](https://arxiv.org/pdf/2209.02646)]
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) [[PDF](https://arxiv.org/pdf/1706.03762)]
  - [Paper walkthrough video](https://www.youtube.com/watch?v=iqmjzecbJHE)
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114v11) [[PDF](https://arxiv.org/pdf/1312.6114v11)]
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) [[PDF](https://arxiv.org/pdf/2207.12598)]
- [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) [[PDF](https://arxiv.org/pdf/1503.03585)]
- [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852) [[PDF](https://arxiv.org/pdf/1502.01852)]
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) [[PDF](https://arxiv.org/pdf/2006.11239)]
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) [[PDF](https://arxiv.org/pdf/2010.02502)]
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09672) [[PDF](https://arxiv.org/pdf/2102.09672)]
  - [(paper walkthrough) DDPM - Diffusion Models Beat GANs on Image Synthesis (Machine Learning Research Paper Explained)](https://www.youtube.com/watch?v=W-O7AZNzbzQ)
- [Lecture Notes in Probabilistic Diffusion Models](https://arxiv.org/abs/2312.10393) [[PDF](https://arxiv.org/pdf/2312.10393)] - Some nice novel images and discussions of diffusion processes
- [State of the Art on Diffusion Models for Visual Computing](https://arxiv.org/abs/2310.07204) [[PDF](https://arxiv.org/pdf/2310.07204)]
- [Stable Diffusion 3 - Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) [[PDF](https://arxiv.org/pdf/2403.03206)]
  - [YouTube paper walkthrough @hu-po](https://youtu.be/yTXMK2TZOZc)
  - [YouTube paper walkthrough @gabrielmongaras](https://youtu.be/6XatajQ-ll0)
- [The Matrix Calculus You Need For Deep Learning](https://arxiv.org/abs/1802.01528) [[PDF](https://arxiv.org/pdf/1802.01528)]
- [Understanding Diffusion Models: A Unified Perspective](https://arxiv.org/abs/2208.11970) [[PDF](https://arxiv.org/pdf/2208.11970)]
- [Variational Diffusion Models](https://arxiv.org/abs/2107.00630) [[PDF](https://arxiv.org/pdf/2107.00630)]

## YouTube Channels

Some noteworthy YouTube channels, some of which have sub-links to relevant ML playlists and individual videos. Note: If a sub-link ends with (V) then it is an individual video rather than a playlist.

- [AI Coffee Break with Letitia](https://www.youtube.com/@AICoffeeBreak/featured)
  - [Diffusion models explained. How does OpenAI's GLIDE work?](https://www.youtube.com/watch?v=344w5h24-h8) (V)
- [Artem Kirsanov](https://www.youtube.com/@ArtemKirsanov/videos)
  - [The Key Equation Behind Probability](https://www.youtube.com/watch?v=KHVR587oW8I) - Entropy, Cross-entropy, KL divergence
  - [The Grandfather Of Generative Models](https://www.youtube.com/watch?v=_bqa_I5hNAo) - Boltzmann Machines
- [3Blue1Brown](https://www.youtube.com/@3blue1brown)
  - [Neural Networks](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) - See the two transformer videos in particular
- [Andrej Karpathy](https://www.youtube.com/@AndrejKarpathy)
  - [CS231n Convolutional Neural Networks for Visual Recognition (2016)](https://www.youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC)
  - [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ)
- [Carnegie Mellon University](https://www.youtube.com/@carnegiemellonuniversityde4339/featured)
  - [11-785 Introduction to Deep Learning](https://deeplearning.cs.cmu.edu/F24/index.html)
  - [Video playlist](https://www.youtube.com/playlist?list=PLp-0K3kfddPwpm8SuB262r4owIkS7NNJj)
- [CUDA Mode](https://www.youtube.com/@CUDAMODE/featured)
- [Computerphile](https://www.youtube.com/@Computerphile/featured)
- [DataScienceCastnet](https://www.youtube.com/@datasciencecastnet/featured) - Johno Whitaker
- [DeepBean](https://www.youtube.com/@deepbean/featured)
  - [Machine Learning](https://www.youtube.com/playlist?list=PLz4ZBoYqCPXrd22wuOquY5ZGzZMk8iT62)
- [ExplainingAI](https://www.youtube.com/@Explaining-AI/featured)
- [Gabriel Mongaras](https://www.youtube.com/@gabrielmongaras/featured) - Some nice research paper walkthroughs.
- [Hamel Husain](https://www.youtube.com/@hamelhusain7140/featured)
  - [Napkin Math For Fine Tuning Pt. 1 w/Johno Whitaker](https://www.youtube.com/watch?v=-2ebSQROew4) (V)
  - [Napkin Math For Fine Tuning Pt. 2 w/Johno Whitaker](https://www.youtube.com/watch?v=u2fJ6K8FjS8) (V)
- [Imperial College London](https://www.youtube.com/@digitallearninghub-imperia3540/featured)
- [Jeremy Howard](https://www.youtube.com/@howardjeremyp/featured)
  - [Practical Deep Learning Part 2](https://www.youtube.com/playlist?list=PLfYUBJiXbdtRUvTUYpLdfHHp9a58nWVXP)
- [Jia-Bin Huang](https://www.youtube.com/@jbhuang0604/featured)
  - [How I Understand Diffusion Models](https://www.youtube.com/watch?v=i2qSxMVeVLI) (V)
  - [How I Understand Flow Matching](https://youtu.be/DDq_pIfHqLs?list=PLdUcsPPD8lGwyjI_wwYYcaNCK4DsEeusM) (V)
- [Kapil Sachdeva](https://www.youtube.com/@KapilSachdeva/featured)
  - [Variational Inference & AutoEncoder](https://www.youtube.com/playlist?list=PLivJwLo9VCUK1dXFU9Ig96fjANOMdoL9R) - Concise and detailed.
  - [KL Divergence - CLEARLY EXPLAINED!](https://www.youtube.com/watch?v=9_eZHt2qJs4)
- [Machine Learning at Berkeley](https://www.youtube.com/@machinelearningatberkeley8868/featured)
  - [CS 198-126: Modern Computer Vision Fall 2022](https://www.youtube.com/playlist?list=PLzWRmD0Vi2KVsrCqA4VnztE4t71KnTnP5)
- [Machine Learning & Simulation](https://www.youtube.com/@MachineLearningSimulation/featured)
- [MIT HAN Lab](https://www.youtube.com/@MITHANLab/featured)
  - [EfficientML.ai Course | 2023 Fall | MIT 6.5940](https://www.youtube.com/playlist?list=PL80kAHvQbh-pT4lCkDT53zT8DKmhE0idB)
- [MIT OpenCourseWare](https://www.youtube.com/@mitocw/playlists)
  - [MIT 18.S096 Matrix Calculus For Machine Learning And Beyond, IAP 2023](https://www.youtube.com/playlist?list=PLUl4u3cNGP62EaLLH92E_VCN4izBKK6OE)
  - [MIT 6.100L Introduction to CS and Programming using Python, Fall 2022](https://www.youtube.com/playlist?list=PLUl4u3cNGP62A-ynp6v6-LGBCzeH3VAQB)
- [MLT Artificial Intelligence](https://www.youtube.com/@MLTOKYO/featured)
- [Outlier](https://www.youtube.com/@outliier/featured)
  - [Diffusion Models | Paper Explanation | Math Explained](https://www.youtube.com/watch?v=HoKDTa5jHvg) (V)
- [Ox educ](https://www.youtube.com/@oxeduc4209/featured)
  - [Bayesian statistics: a comprehensive course](https://www.youtube.com/playlist?list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm)
- [Ritvikmath](https://www.youtube.com/@ritvikmath/featured)
  - [Data Science Basics](https://www.youtube.com/playlist?list=PLvcbYUQ5t0UG5v62E_QO7UihkfePakzNA)
  - [Data Science Concepts](https://www.youtube.com/playlist?list=PLvcbYUQ5t0UH2MS_B6maLNJhK0jNyPJUY)
- [Sam Witteveen](https://www.youtube.com/@samwitteveenai/featured)
  - [RAG - Retrieval Augmented Generation](https://www.youtube.com/playlist?list=PL8motc6AQftn-X1HkaGG9KjmKtWImCKJS)
- [San Diego Machine Learning](https://www.youtube.com/@SanDiegoMachineLearning/featured)
  - [Understanding Deep Learning](https://www.youtube.com/playlist?list=PLmp4AHm0u1g0AdLp-LPo5lCCf-3ZW_rNq) - Book study group
- [Stanford Online](https://www.youtube.com/@stanfordonline/featured)
  - [CS109 Introduction to Probability for Computer Scientists](https://web.stanford.edu/class/cs109/)
    - [Video playlist](https://www.youtube.com/playlist?list=PLoROMvodv4rOpr_A7B9SriE_iZmkanvUg)
    - [Course reader](https://chrispiech.github.io/probabilityForComputerScientists/en/index.html)
  - [Stanford CS224N: NLP with Deep Learning](https://www.youtube.com/playlist?list=PLoROMvodv4rMFqRtEuo6SGjY4XbRIVRd4) - Stanford course on transformers with review sessions on Python, PyTorch, and Hugging Face.
  - [Stanford CS236: Deep Generative Models I 2023 I Stefano Ermon](https://www.youtube.com/playlist?list=PLoROMvodv4rPOWA-omMM6STXaWW4FvJT8)
    - [Course website](https://deepgenerativemodels.github.io/)
- [StatQuest](https://www.youtube.com/@statquest) - Highly recommended for probability and statistics.
  - [Machine Learning](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF)
  - [Statistics Fundamentals](https://www.youtube.com/playlist?list=PLblh5JKOoLUK0FLuzwntyYI10UQFUhsY9)
- [Tanishq Abraham](https://www.youtube.com/@tanishqabraham3419/featured)
  - [Diffusion Models Study Group](https://www.youtube.com/playlist?list=PLXqc0KMM8ZtKVEh8fIWEUaIU43SmWnfdM) - Study group discussing many popular ML papers.
- [Tübingen Machine Learning](https://www.youtube.com/c/T%C3%BCbingenML/featured)
  - [Math for Deep Learning — Andreas Geiger](https://www.youtube.com/playlist?list=PL05umP7R6ij0bo4UtMdzEJ6TiLOqj4ZCm)
  - [Probabilistic Machine Learning -- Philipp Hennig, 2023](https://www.youtube.com/playlist?list=PL05umP7R6ij2YE8rRJSb-olDNbntAQ_Bx)
  - [Statistical Machine Learning — Ulrike von Luxburg, 2020](https://www.youtube.com/playlist?list=PL05umP7R6ij2XCvrRzLokX6EoHWaGA2cC)
- [TWIML Community](https://www.youtube.com/@TWIMLCommunity/featured)
  - [Practical DL for Coders Part 2 Study Group](https://www.youtube.com/playlist?list=PLesM8TI75z-HifxCo34zF4pHcyIRPV1Ko)
- [Umar Jamil](https://www.youtube.com/@umarjamilai/featured)
  - [Attention is all you need (Transformer) - Model explanation (including math), Inference and Training](https://youtu.be/bCz4OMemCcA)
  - [How diffusion models work - explanation and code!](https://www.youtube.com/watch?v=I1sPXkm2NH4)
  - [Variational Autoencoder - Model, ELBO, loss function and maths explained easily!](https://youtu.be/iwEzwTTalbg)
- [Volodymyr Kuleshov (Cornell Tech)](https://www.youtube.com/@vkuleshov/featured)
  - [Applied Machine Learning (Cornell Tech CS 5787, Fall 2020)](https://www.youtube.com/playlist?list=PL2UML_KCiC0UlY7iCQDSiGDMovaupqc83)
  - [Deep Generative Models (Cornell Tech CS 6785, Spring 2023)](https://www.youtube.com/playlist?list=PL2UML_KCiC0UPzjW9BjO-IW6dqliu9O4B)
- [Weights & Biases](https://www.youtube.com/@WeightsBiases/featured)
  - [W&B Fastbook Reading Group](https://www.youtube.com/playlist?list=PLD80i8An1OEHdlrBwa7mKFaHX9tH86b93) (Covers part1 of the fastai course)
- [Yannic Kilcher](https://www.youtube.com/@YannicKilcher/featured)

## Individual videos.

Collection of one-off individual videos.

### Training & Fine Tuning

- [Napkin Math For Fine Tuning](https://x.com/HamelHusain/status/1798353336145674483) - Video from Johno on fine tuning.

### Transformers

- [Transformers with Lucas Beyer, Google Brain](https://www.youtube.com/watch?v=EixI6t5oif0)

### Diffusion

- [CVPR #18546 - Denoising Diffusion Models: A Generative Learning Big Bang](https://www.youtube.com/watch?v=1d4r19GEVos)
- [Diffusion and Score-Based Generative Models](https://www.youtube.com/watch?v=wMmqCMwuM2Q) (Yang Song)
- [Evidence Lower Bound (ELBO) - Clearly Explained!](https://youtu.be/IXsA5Rpp25w)
- [Stable Diffusion Explained w/ Sai Kumar](https://www.youtube.com/watch?v=V3zBHGB0LWs)
- [Kullback–Leibler divergence (KL divergence) intuitions](https://www.youtube.com/watch?v=NPGkSvCJBzc) - Intuitive breakdown of the KL divergence equation.
- [MIT 6.S191: Deep Generative Modeling](https://youtu.be/Dmm4UG-6jxA?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI)
- [The Reparameterization Trick](https://youtu.be/vy8q-WnHa9A)

## Podcasts

- [Gradient Dissent - A Machine Learning Podcast](https://www.youtube.com/playlist?list=PLD80i8An1OEEb1jP0sjEyiLG8ULRXFob_)
- [Latent Space](https://www.youtube.com/playlist?list=PLWEAb1SXhjlfkEF_PxzYHonU_v5LPMI8L)
- [No Priors Podcast](https://www.youtube.com/playlist?list=PLMKa0PxGwad7jf8hwwX8w5FHitXZ1L_h1)

## Newsletters

- [Data Science Weekly Newsletter](https://datascienceweekly.substack.com/)

## Courses
- [Class Central](https://www.classcentral.com/) - Find courses from top Universities and companies.
- [Fast.ai](https://www.fast.ai/)
  - [Deep Learning from the Foundations](https://course19.fast.ai/index.html) (2019)
  - [Practical Deep Learning for Coders](https://course.fast.ai/) (2022)
- [Hugging Face Diffusion Models Course](https://huggingface.co/learn/diffusion-course/en/unit0/1) - Comprehensive course on diffusion models from Hugging Face.
  - [GitHub Repository](https://github.com/huggingface/diffusion-models-class)
    - [Unit 1 walkthrough (with Johno)](https://www.youtube.com/watch?v=09o5cv6u76c)
    - [Unit 2 walkthrough (with Johno)](https://www.youtube.com/watch?v=mY20iKOQ2zw)
- [Introduction to Deep Learning (MIT)](http://introtodeeplearning.com/) - MIT course on deep learning, including a video on deep generative modeling.
  - [Video playlist](https://www.youtube.com/playlist?list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI)
- [Machine Learning University](https://aws.amazon.com/machine-learning/mlu/)
- [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
- [Reddit Thread on Advanced Courses](https://www.reddit.com/r/MachineLearning/comments/fdw0ax/d_advanced_courses_update/) - Discussion on advanced machine learning courses.
- [Stable Diffusion Implementation](https://github.com/DrugowitschLab/ML-from-scratch-seminar/tree/master/StableDiffusion) - Implementation of Stable Diffusion from scratch.
- [Stanford University](https://www.stanford.edu/)
  - [CS149 PARALLEL COMPUTING](https://gfxcourses.stanford.edu/cs149/fall23/)

## Books

There are many excellent textbooks available for machine learning and deep learning. Here are a few that I've come acrsoos that are particularly useful. Some of these have free online versions available in PDF or HTML format, and links to these where available are added after the book title.

- [Deep Learning (Goodfellow, Bengio, Courville)](https://www.amazon.com/Deep-Learning-Adaptive-Computation-Machine/dp/0262035618) [[HTML](https://www.deeplearningbook.org/)]
- [Deep Learning, A Visual Approach (Glassner)](https://www.glassner.com/portfolio/deep-learning-a-visual-approach/)
  - [GitHub repository for the book containing figures, errata, and notebooks](https://github.com/blueberrymusic/Deep-Learning-A-Visual-Approach)
- [Dive into Deep Learning](https://d2l.ai/index.html) [[HTML](https://d2l.ai/chapter_preface/index.html)] - Interactive deep learning textbook with code, math, and discussions.
- [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering?tab=readme-ov-file) - This is the GitHub repo for the book. A PDF is also available.
- [Mathematics for Machine Learning](https://www.amazon.com/Mathematics-Machine-Learning-Peter-Deisenroth/dp/110845514X/) [[PDF](https://mml-book.github.io/book/mml-book.pdf)]
  - [Official book website](https://mml-book.github.io/external.html)
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html) - Free online book on neural networks and deep learning.
- [Probabilistic Machine Learning](https://probml.github.io/pml-book/) (set of three books)
- [Probability and Statistics - The Science of Uncertainty 2e](https://www.utstat.toronto.edu/mikevans/jeffrosenthal/) [[PDF]](https://www.utstat.toronto.edu/mikevans/jeffrosenthal/book.pdf) [[Solutions manual]](https://www.utstat.toronto.edu/mikevans/jeffrosenthal/EvansRosenthalsolutions.pdf)
- [Python for Data Analysis, 3E](https://wesmckinney.com/book/) - Free online version available
- [Rectified Flow](https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html)
- [Understanding Deep Learning](https://udlbook.github.io/udlbook/) [[PDF](https://github.com/udlbook/udlbook/releases/download/v4.0.2/UnderstandingDeepLearning_07_02_24_C.pdf)]
  - [Jupyter Notebooks from the book](https://github.com/udlbook/udlbook/tree/main/Notebooks)
  - [Selected solutions to Jupyter Notebooks](https://github.com/total-expectation/udlbook/tree/main/Notebooks)

### Python Books

- [Automate the Boring Stuff with Python](https://automatetheboringstuff.com/)

## Blogs

A collection of blogs and posts that post about various ML topics. Sub-links are individually selected posts from the related (parent) blog.

- [Berkeley Artificial Intelligence Research](https://bair.berkeley.edu/blog/)
- [Cambridge MLG](https://mlg.eng.cam.ac.uk/blog/)
  - [An introduction to Flow Matching](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html)
- [Chris Levy](https://drchrislevy.github.io/blog.html)
- [Google DeepMind](https://deepmind.google/discover/blog/)
- [Distill](https://distill.pub/)
  - [Feature Visualization](https://distill.pub/2017/feature-visualization/)
- [Hugging Face](https://huggingface.co/blog)
  - [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion)
- [Isamu Isozaki](https://isamu-website.medium.com/)
  - [Understanding Diffusion Models Part 1](https://isamu-website.medium.com/understanding-diffusion-models-to-generate-my-favorite-dog-part-1-4be07ec0fc29)
  - [Understanding Diffusion Models Part 2](https://isamu-website.medium.com/understanding-diffusion-models-to-generate-my-favorite-dog-part-2-4ad76a28c2a4)
- [Jack Tol](https://jacktol.net/)
- [Jake Tae](https://jaketae.github.io/categories/)
- [Jay Alammar](https://jalammar.github.io/)
  - [Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual guide to understanding transformer architectures.
  - [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/)
- [Jeremy Jordan](https://www.jeremyjordan.me/data-science/)
  - [Variational Autoencoders Explained](https://www.jeremyjordan.me/variational-autoencoders/) - Detailed explanation of VAEs with intuitive breakdowns.
- [Kapil Sachdeva](https://ksachdeva17.medium.com/)
  - [But what is Entropy?](https://towardsdatascience.com/but-what-is-entropy-ae9b2e7c2137)
  - [But what is a Random Variable?](https://towardsdatascience.com/but-what-is-a-random-variable-4265d84cb7e5)
- [Lilian Weng](https://lilianweng.github.io/archives/)
  - [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
- [Matthew N. Bernstein](https://mbernste.github.io/year-archive/) - Some gems here on various diffusion related topics
- [MIT News - AI](https://news.mit.edu/topic/artificial-intelligence2)
- [OpenAI](https://openai.com/news/research/)
- [Paperspace](https://blog.paperspace.com/)
- [Probably Overthinking It](https://www.allendowney.com/blog/) - Some really nice posts on general probability topics.
  - [Density and Likeihood: What's the Difference?](https://www.allendowney.com/blog/2024/07/13/density-and-likelihood/)
- [Radek Osmulski](https://radekosmulski.medium.com/)
  - [A practitioner's guide to PyTorch](https://towardsdatascience.com/a-practitioners-guide-to-pytorch-1d0f6a238040)
- [Reddit](https://www.reddit.com/)
  - [r/MachineLearning](https://www.reddit.com/r/MachineLearning/)
  - [r/StableDiffusion](https://www.reddit.com/r/StableDiffusion/)
    - [Different sampling methods](https://www.reddit.com/r/StableDiffusion/comments/zgu6wd/comment/izkhkxc/?context=3)
	- [Generating consistent characters without an embedding, hypernetwork, or Dreambooth model](https://www.reddit.com/r/StableDiffusion/comments/10h0aud/generating_consistent_characters_without_an/)
    - [My attempt to explain Stable Diffusion at a ELI15 level](https://www.reddit.com/r/StableDiffusion/comments/xm7ndc/my_attempt_to_explain_stable_diffusion_at_a_eli15/)
	- [Simple trick I use to get consistent characters in SD](https://www.reddit.com/r/StableDiffusion/comments/zv5mbq/simple_trick_i_use_to_get_consistent_characters/)
    - [Steps for getting better images](https://www.reddit.com/r/StableDiffusion/comments/x6kjdh/steps_for_getting_better_images/)
  - [(old) reddit Stable Diffusion](https://old.reddit.com/r/StableDiffusion/)
    - [Beginner/Intermediate Guide to Getting Cool Images from Stable Diffusion](https://old.reddit.com/r/StableDiffusion/comments/x41n87/how_to_get_images_that_dont_suck_a/)
- [Salman Naqvi](https://forbo7.github.io/forblog/)
- [Sander Dieleman](https://sander.ai/posts/) - Collection of posts on various advanced ML topics.
  - [Perspectives on diffusion](https://sander.ai/2023/07/20/perspectives.html) - Insights on diffusion models.
  - [Diffusion models are autoencoders](https://sander.ai/2022/01/31/diffusion.html)
  - [Geometry of Diffusion Models](https://sander.ai/2023/08/28/geometry.html) - Exploration of the geometric aspects of diffusion models.
- [Scratchapixel - Welcome to Computer Graphics](https://www.scratchapixel.com/index.html) (general computer graphics programming)
- [Stable Diffusion Art](https://stable-diffusion-art.com)
  - [Stable Diffusion Samplers: A Comprehensive Guide](https://stable-diffusion-art.com/samplers/)
- [Stability.ai](https://stability.ai/news)
- [The Latent: Code the Maths](https://magic-with-latents.github.io/latent/archive.html) - 3 blog posts on DDPMs
- [Vishal Bakshi](https://vishalbakshi.github.io/blog/)
  - [How Does Stable Diffusion Work?](https://vishalbakshi.github.io/blog/posts/2024-08-08-how-does-stable-diffusion-work/)
  - [Paper Summary: Attention is All You Need](https://vishalbakshi.github.io/blog/posts/2024-03-30-attention-is-all-you-need/)

## Blog Posts

Some interesting and useful individual blog posts.

- [Classification loss function as comparing 2 vectors](https://dienhoa.github.io/dhblog/posts/classification_loss_func.html) - Intuition of cross-entropy viewed as comparing 2 vectors. 
- [Diffusion Models - Wiki](https://en.wikipedia.org/wiki/Diffusion_model)
- [Diffusion Models From Scratch](https://www.tonyduan.com/diffusion/index.html) - In-depth tutorial covering derivations and core concepts of diffusion models.
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/) - Detailed blog post on score-based generative models and their advantages.
- [Introduction to Attention Mechanism](https://erdem.pl/2021/05/introduction-to-attention-mechanism)
- [Understanding PyTorch with an example: a step-by-step tutorial](https://towardsdatascience.com/understanding-pytorch-with-an-example-a-step-by-step-tutorial-81fc5f8c4e8e)
- [The ELBO in Variational Inference](https://gregorygundersen.com/blog/2021/04/16/variational-inference/)
- [Step by Step visual introduction to Diffusion Models](https://erdem.pl/2023/11/step-by-step-visual-introduction-to-diffusion-models)
- [What is a variational autoencoder?](https://jaan.io/what-is-variational-autoencoder-vae-tutorial/)

## GitHub Repositories

Some tutorials and other resources located on GitHub.

- [Denoising diffusion probabilistic models](https://github.com/acids-ircam/diffusion_models/tree/main)
  - [Diffusion probabilistic models - Score matching](https://github.com/acids-ircam/diffusion_models/blob/main/diffusion_01_score.ipynb)
  - [Diffusion probabilistic models - Introduction](https://github.com/acids-ircam/diffusion_models/blob/main/diffusion_02_model.ipynb)

### Student Repos

This is a collection of GitHub repositories of students of the Fasti.ai course (parts 1 & 2).

- [math-fastai](https://github.com/total-expectation/math-fastai) - This repository includes some really nice probability notebooks.

### Code Snippets and Examples

- [PyTorch VAE Python implementation](https://github.com/pytorch/examples/blob/main/vae/main.py)

## Discord Channels

- Fastai
- Hugging Face
- Stable Diffusion

## Software and Tools

- [Answer.ai](https://www.answer.ai/)
  - [AnswerDotAI (GitHub)](https://github.com/orgs/AnswerDotAI/repositories)
    - [claudette](https://github.com/AnswerDotAI/claudette)
- [Eureka Labs](https://eurekalabs.ai/)
- [Fast.ai](https://docs.fast.ai/)
- [Hugging Face](https://huggingface.co/)
  - [Diffusers tutorials](https://huggingface.co/docs/diffusers/en/index)
  - [General documentation](https://huggingface.co/docs)
  - [Hugging Face Spaces](https://huggingface.co/spaces) - Host and demo models
- [nbdev](https://nbdev.fast.ai/)
- [PyTorch](https://pytorch.org/)
  - [Tutorials](https://pytorch.org/tutorials/)
- Claudette (Answer.ai)
- Jupyter
- Zotero
- [Gradio](https://www.gradio.app/)
- Streamlit
- Cursor
- LightningAI
- [Kaggle](https://www.kaggle.com/)
- [Fal](https://fal.ai/)

## Cloud Services

Various services for hosting and deploying deep learning models.

- Replicate
- Paperspace
- LightningAI
- [Paperspace](https://www.paperspace.com/)
- [Lambda](https://lambdalabs.com/)
- [Jarvislabs](https://jarvislabs.ai/)
- [Vast.ai](https://vast.ai/)
- [OctoAI](https://octo.ai)
- [Anthropic ](https://www.anthropic.com/)
- [Answer.AI](https://www.answer.ai/)
- [Qwak](https://www.qwak.com/) - Now part of JFrog

## Datasets

- [LAION](https://laion.ai/)

## Twitter

Machine learning on Twitter is very active. Here are some of the people you should be following!

- [Tanishq Abraham - @iScienceLuvr](https://x.com/iScienceLuvr)
- [Jeremy Howard - @jeremyphoward](https://x.com/jeremyphoward)
- [Andrej Karpathy - @karpathy](https://x.com/karpathy)
- [Jonathan Whitaker - @johnowhitaker](https://x.com/johnowhitaker)

## Interviews

- [Prof. Chris Bishop's NEW Deep Learning Textbook!](https://youtu.be/kuvFoXzTK3E)
- [This is why Deep Learning is really weird (Prof. Simon Prince)](https://youtu.be/sJXn4Cl4oww)

## Jupyter Notebooks

- [Diffusion Models from Scratch](https://colab.research.google.com/github/huggingface/diffusion-models-class/blob/main/unit1/02_diffusion_models_from_scratch.ipynb)

## Other Resource Lists

- [Awesome-ChatGPT](https://github.com/awesome-gptX/awesome-gpt)

## Additional Resources

- [Awesome Generative AI Guide](https://github.com/aishwaryanr/awesome-generative-ai-guide) - Curated list of generative AI resources, courses, and materials.
- [Experiments with Google (deprecated as of 2022)](https://experiments.withgoogle.com/) - Showcasing various AI experiments.
- [Hacker News Showcase](https://news.ycombinator.com/show) - People showcasing their projects.
- [Labs.Google](https://labs.google/) - Experiment with the future of AI.
- [ML Ops Guide (Chip Huyen)](https://huyenchip.com/mlops/) - Collection of resources for machine learning operations.
- [Multi-AI Agent Systems with CrewAI](https://www.deeplearning.ai/short-courses/multi-ai-agent-systems-with-crewai/) - Course on building multi-agent AI systems.
- [Papers With Code](https://paperswithcode.com/)
- [Product Hunt](https://www.producthunt.com/) - Inspiration for some AI products others have launched.
- [Roboflow object detection in sports](https://github.com/roboflow/sports)
- [Find Trending Papers](https://trendingpapers.com)
- [UNet Diffusion Model in CUDA](https://github.com/clu0/unet.cu) - Implementation of a UNet diffusion model in pure CUDA.
- [VAE Slides](https://deeplearning.cs.cmu.edu/S22/document/slides/lec21.VAE.pdf) - Lecture slides on Variational Autoencoders.
- [The Bright Side of Mathematics](https://thebrightsideofmathematics.com/)
- [MLOps guide](https://huyenchip.com/mlops/) - Chip Huyen
- [ghapi](https://ghapi.fast.ai/) - GitHub API from fast.ai