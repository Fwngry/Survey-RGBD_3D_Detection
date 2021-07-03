# Deepcontext - 三维上下文

三维上下文已被证明对场景理解极为重要，但在将上下文信息与深度神经网络架构整合方面的研究却很少。

本文提出了一种将三维上下文嵌入神经网络拓扑结构的方法，该网络经过训练可以进行整体的场景理解。给定一个描述三维场景的深度图像，我们的网络将观察到的场景与预先设定的三维场景模板对齐，然后推理出场景模板中每个物体的存在和位置。

这样一来，我们的模型就能在一个三维卷积神经网络的单一前向通道中识别多个物体，同时捕捉到全局场景和局部物体信息。为了创建这个三维网络的训练数据，我们生成了部分合成的深度图像，这些图像是通过用相同物体类别的CAD模型库替换真实物体而呈现的1。

> 过程：根据深度图像，将观察到的场景与预设场景对齐，推理出obj的存在和位置
>
> 单一向前，识别多个物体
>
> 使用合成数据作为训练数据，具有相对较强上下文证据
>
> 我们表明，上下文模型提供了与本地物体检测器互补的信息，这可以很容易地整合。
>
> 情境模型与本地物体检测器DSS[34]的比较。对于深度缺失的物体（第1、3行的显示器）、严重遮挡的物体（第2行的床头柜）效果很好，可以防止检测到错误的排列（DSS结果中错误的桌子和床头柜）。

![image-20210626183744463](https://oj84-1259326782.cos.ap-chengdu.myqcloud.com/uPic/2021/06_26_image-20210626183744463.png)

![image-20210701112533739](https://oj84-1259326782.cos.ap-chengdu.myqcloud.com/uPic/2021/07_01_image-20210701112533739.png)

pipeline：

1. 生成模版
2. 考虑到来自深度图像的三维体积输入，我们首先将场景模板与输入数据对齐 Transformation network。
3. 考虑到initial alignment，我们的3D Context network估计物体的存在；
4. 并根据本地物体特征和整体场景特征调整物体的位置，以产生最终的三维场景理解结果。

## 一、引言

在三维空间中理解室内场景在许多应用中是非常有用的，如室内机器人技术、增强现实。为了支持这项任务，本文的目标是从单一深度图像中识别家具的类别和三维位置。

在以前的许多工作中，上下文已被成功地用于处理这一具有挑战性的问题。特别是整体的场景上下文模型，它整合了自下而上的局部证据和自上而下的场景上下文，已经取得了卓越的性能[6, 23, 24, 48, 49]。**然而，它们有一个严重的缺点，即自下而上和自上而下的阶段是分开运行的。自下而上的阶段只使用局部证据，需要产生大量的噪声假设以确保高召回率，而自上而下的推理通常需要组合算法，如信念传播或MCMC，这些算法的计算量很大，整个组合系统很难高效和稳健地实现一个合理的最优解。**

受深度学习成功的启发，我们提出了一种三维深度卷积神经网络架构，它能有效地联合利用局部外观和全局场景背景来理解三维场景。

**挑战：不能用固定维度来描述上下文信息。**

**设计一个深度学习架构来编码场景理解的上下文是具有挑战性的。不同于物体的位置和大小可以用固定数量的参数来表示，场景可能涉及未知数量的物体，因此需要可变维度来表示，这很难用固定架构的卷积神经网络来实现。**

此外，尽管整体场景模型允许灵活的上下文，但它们需要共同的知识来手动预设物体之间的关系，例如床和床头柜之间的相对距离。因此，该模型可能会不必要地编码薄弱的上下文，忽略重要的上下文，或以过于简单的方式测量上下文。

**解决：不存在的预测，实现固定维度。**

为了解决这些问题，我们提出并学习一个用场景模板编码的场景表示法。**场景模板包含了一组具有很强上下文关联性的物体，这些物体可能会出现在一个具有相对约束的家具布置的场景中。它允许对所涉及的物体进行 "不存在 "的预测，因此各种场景可以用一个固定的维度来表示。**一个场景可以被看作是一个激活了物体子集的场景模板。场景模板也会学习只考虑有强背景的物体，我们认为无背景的物体，如椅子可以任意放置，应该由基于局部外观的物体检测器来检测。



**一个模版，等价于一个功能子区域。在场景中，预设了布置以及物体在场景中的位置。**

每个模板代表了室内场景的一个功能子区域，预设了典型的家具布置和可能的物体相对于模板参考框架的估计三维锚点位置。

**1. 转换网络：用于场景与模版对齐，将模版锚，作为神经网络的先验因素**

我们通过设计一个转换网络，将这些模板锚作为神经结构的先验因素，使输入的三维场景（对应于观察到的深度图像）与模板（即三维空间中的典型家具布置）对齐。

**2. 三维背景神经网络：场景被送入网络，全局整体+感兴趣局部；分类+回归位置**

然后，对齐的三维场景被送入一个三维背景神经网络，以确定场景模板中每个物体的存在和位置。这个三维上下文神经网络包含一个整体的场景路径和一个使用三维兴趣区域（ROI）集合的物体路径，以便分别对物体的存在进行分类并回归物体的位置。我们的模型学会了利用两条路径的全局和局部信息，并能在三维神经网络的**一次正向传递中识别多个物体**。值得注意的是，我们没有手动定义物体之间的上下文关系，而是允许网络以任意格式自动学习所有物体的上下文。

**挑战：需要大量数据进行训练；解决：CAD在场景中该队物体进行替换并原地渲染。**

数据是训练我们网络的另一个挑战性问题。整体的场景理解需要3D ConvNet有足够的模型容量，这需要用**大量的数据进行训练**。然而，现有的用于场景理解的RGB-D数据集都很小。**为了克服这一限制，我们从现有的RGB-D数据集中合成训练数据，将场景中的物体替换为来自同一物体类别的CAD模型库中的物体，并在原地渲染以生成部分合成的深度图像。**我们的合成数据表现出各种不同的局部物体外观，同时仍然保持真实场景中的室内家具排列和杂乱。

在实验中，我们用这些合成数据对我们的网络进行预训练，然后在少量的真实数据上进行ﬁnetune，而同样的网络在真实数据上进行di-rective训练后，就不能再对其进行ﬁnetune。

而同样的网络在真实数据上的训练却无法收敛。

本文的贡献主要体现在三个方面。

1）我们提出了一种场景模板表示，使我们能够使用深度学习方法来理解场景和学习背景。场景模板只对具有强背景的物体进行编码，并为场景家族提供了一个**固定维度的表示**。

2）我们提出了一个三维上下文神经网络，可以自动学习场景上下文。它同时利用全局背景和局部外观，并在网络的**一次前向传递中有效地检测出所有处于背景中的物体。**

3）我们提出了**一种混合数据增强方法，该方法生成的深度图像保持了真实场景中的室内家具排列**，但包含具有不同外观的合成物体。

## 二、Overview

我们的方法是先从训练数据中自动构建一套场景模板（见3.1节）。

**（模版=特定功能的子区域）**每个场景模板不是对场景中所有事物的整体模型，而是只代表在一个场景的子区域中执行特定功能的有背景的物体。每个模板定义了一个或多个不同物体类别的实例在一个固定维度上的可能布局的分布。

给出一个场景的深度图作为输入2，我们将其转换为场景的三维体积表示，并将其送入神经网络。

**神经网络首先推断出适合表示该场景的场景模板：**

**A：如果没有一个预设的场景模板被满足，则将其留给基于本地外观的物体检测器。**

**B：如果选择了一个场景模板，变换网络就会估计出旋转和平移，使场景与推断出的场景模板对齐。**

有了这个初始对齐，三维上下文网络就会提取编码场景上下文的全局场景特征和模板中定义的每个锚点物体的局部物体特征，这些特征被**串联在一起**，以预测模板中每个锚点物体的存在。并通过偏**移量**来调整其边界框，以达到更好的物体识别度。

最后的结果是对场景的理解，包括场景中每个物体的三维位置和类别，以及房间的布局元素，包括墙壁、地板和天花板，它们在网络中被表示为物体。

## 三、Scene Template

数据驱动的模版 - 定义

基于GT生成模版

## 四、三维场景分析网络

给定一个深度图像作为输入，我们首先使用截断符号距离函数（TSDF）[34, 28]将其转换为三维体积表示。我们使用128×128×64的TSDF网格来包括整个场景，体素单位大小为0.05米，截断值为0.15米。这个TSDF表示被送入三维神经网络，使模型在三维空间自然运行并直接产生三维输出。

场景模版分类网络 - Scene Template Classification Network

我们首先训练一个神经网络来估计输入场景的场景模板类别（图3，场景路径）。输入场景的TSDF表示首先被送入3层3D卷积+3D池化+ReLU，并转换为空间特征图。在通过两个全连接层后，三维空间特征被转换为一个全局特征向量，编码整个场景的信息。全局特征被用于经典的softmax层的场景模板分类。在测试过程中，如果置信度足够高（>0.95），我们会选择对输入场景得分最高的场景模板。否则，我们就不运行我们的方法，因为没有一个场景模板符合输入场景的要求。这样的场景会被传递给基于局部外观的物体检测器进行物体检测。在实践中，这四个场景模板可以与SUN-RGBD数据集中从各种室内环境中捕获的一半以上的图像相匹配。

We ﬁrst train a neural network to estimate the scene template category for the input scene (Fig. 3, Scene pathway). The TSDF representation of the input scene is ﬁrstly fed into 3 layers of 3D convolution + 3D pooling + ReLU, and converted to a spatial feature map. After passing through two fully connected layers, the 3D spatial feature is converted to a global feature vector that encodes the information from the whole scene. The global feature is used for scene template classiﬁcation with a classic softmax layer. During testing, we choose the scene template with the highest score for the input scene if the conﬁdence is high enough (> 0.95). Otherwise, we do not run our method because none of the scene templates ﬁts the input scene. Such scenes are passed to a local appearance based object detector for object detection. In practice, the four scene templates can match with more than half of the images in the SUN-RGBD dataset captured from various of indoor environments.

转换网络 - Transformation network

鉴于场景模板的类别，我们的方法估计了一个由三维旋转和平移组成的全局变换，将输入场景的点云与目标预设的场景模板对齐（图4）。这实质上是一个将输入场景中的主要物体与场景模板中的物体对齐的变换。这使得这一阶段的结果对输入的旋转不产生影响，并且物体的墙和边界框被全局地对齐到三个主要方向。我们架构的下一部分，即三维上下文网络，依靠这种对齐方式来获得物体的方向和基于场景模板的三维物体锚点位置的集合特征。

我们首先估计旋转。我们假设重力方向是给定的，例如来自加速度计。在我们的案例中，这个重力方向是由我们实验中使用的SUN RGB-D数据集提供的。因此，我们只需要估计偏航，将输入点云在水平面上旋转到图1所示的场景模板视点。我们将360度的旋转范围划分为36个仓，并将这个问题归纳为一个分类任务（图4）。我们使用与第4.1节中介绍的场景模板分类网络相同的架构来训练3D ConvNet，除了为softmax生成一个36通道的输出。在训练过程中，我们将每个训练输入场景对准点云的中心，并为旋转（+/-10度）和翻译（点云范围的1/6）添加噪音。

对于平移，我们应用相同的网络结构，在应用预测的旋转后识别平移。我们的目标是预测输入点云的主要物体的中心和其相应的场景模板之间的三维偏移。为了实现这一目标，我们将三维翻译空间离散成一个0.5米3分辨率的网格，其尺寸为[-2.5, 2.5]×[-2.5, 2.5]×[-1.5, 1]，并将这一任务再次表述为一个726路分类问题（图4）。我们尝试用各种损失函数进行直接回归，但效果不如分类法好。我们还尝试了一种基于ICP的方法，但它不能产生良好的效果。

三维内容网络 - 3D Context network

现在我们描述一下使用场景模板进行室内场景解析的上下文神经网络。对于上一节中定义的每个场景模板，都要训练一个单独的预测网络。如图3所示，该网络有两条路径。

全局场景路径，给定一个与模板对齐的坐标系中的三维体积输入，产生一个保留了输入数据中空间结构的空间特征和整个场景的全局特征。

对于物体路径，我们把来自场景路径的空间特征图作为输入，并根据特定物体的三维场景模板汇集本地的三维感兴趣区域（ROI）。

三维兴趣区域池是一个6×6×6分辨率的最大池，灵感来自于[9]的二维兴趣区域池。三维汇集的特征然后通过两层三维卷积+三维汇集+ReLU，然后与来自场景路径的全局特征向量相连接。再经过两个全连接层，网络预测物体的存在（二元分类任务）以及与第3.1节中学习的锚点位置相关的三维物体边界框（三维位置和尺寸）的偏移（使用L1smooth loss[34]的回归任务）。在物体特征向量中包括全局场景特征向量，提供整体的背景信息，以帮助识别物体是否存在及其位置。

We now describe the context neural network for indoor scene parsing using scene templates. For each scene template deﬁned in the previous section, a separate prediction network is trained. As shown in Fig. 3, the network has two pathways. The global scene pathway, given a 3D volumetric input in a coordinate system that is aligned with the template, produces both a spatial feature that preserves the spatial structure in the input data and a global feature for the whole scene. For the object pathway, we take the spatial feature map from the scene pathway as input, and pool the local 3D Region Of Interest (ROI) based on the 3D scene template for the speciﬁc object. The 3D ROI pooling is a max pooling at 6×6×6 resolution, inspired by the 2D ROI pooling from [9]. The 3D pooled features are then passed through 2 layers of 3D convolution + 3D pooling + ReLU, and then concatenated with the global feature vector from the scene pathway. After two more fully connected layers, the network predicts the existence of the object (a binary classiﬁcation task) as well as the offset of the 3D object bounding box (3D location and size) related to the anchor locations learned in Sec. 3.1 (a regression task using L1smooth loss [34]). Including the global scene feature vector in the object feature vector provides holistic context information to help identify if the object exists and its location.

训练

## 五、混合数据生成 - 预训练

## 六、实验

![image-20210626185713276](https://oj84-1259326782.cos.ap-chengdu.myqcloud.com/uPic/2021/06_26_06_26_image-20210626185713276.png)

