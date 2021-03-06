> 原文地址 [blog.csdn.net](https://blog.csdn.net/weixin_36670529/article/details/101390879)

> RGBD输入，3D框输出
>
> 数据集：SUN-RGBD
>
> 2D驱动3D，Frustum鼻祖，快得多，解决了DSS全局搜索和CAD类别限制导致的慢速问题
>
> pipeline：
>
> 1. 2D 目标检测方法定位可能的目标周围的 2D 边界框。每个 2D 边界框在 3D 中扩展到我们所说的截锥体。（2D信息连续、更快、固定区域搜索）
> 2. 我们估计场景和单个目标的方向，其中每个目标都有自己的方向。
> 3. 我们训练了一个多层感知器，利用方向上的点密度，在每个方向上回归三维物体的边界。
> 4. 基于目标类再现和类间距离的上下文信息细化检测分数。

摘要
==

在本文中，我们提出了一种在 RGB-D 场景中，在目标周围放置三维包围框的技术。

1. 我们的方法充分利用二维信息，利用最先进的二维目标检测技术，快速减少三维搜索空间。
2. 然后，我们使用 3D 信息来定位、放置和对目标周围的包围框进行评分。我们使用之前利用常规信息的技术，独立地估计每个目标的方向。
3. 三维物体的位置和大小是用多层感知器 (MLP) 学习的。
4. 在最后一个步骤中，我们根据场景中的目标类关系改进我们的检测。

最先进的检测方法相比，操作几乎完全在稀疏的 3D 域, 在著名的 SUN RGB-D 实验数据集表明, 我们建议的方法要快得多 (4.1 s / 图像)RGB-D 图像中的 3 目标检测和执行更好的地图(3) 高于慢是 4.7 倍的最先进的方法和相对慢两个数量级的方法。这一工作提示我们应该进一步研究 3D 中 2D 驱动的目标检测，特别是在 3D 输入稀疏的情况下。

1、简介
====

场景理解的一个重要方面是目标检测，它的目标是在对象周围放置紧密的 2D 边界框，并为它们提供语义标签。2D 目标检测的进步是由众多挑战中令人印象深刻的性能驱动的，并由具有挑战性和大规模的数据集支持。

0. 二维目标检测技术的进步体现在快速、准确检测技术的发展和普及。由于二维目标检测结果受图像帧的约束，需要更多的信息将其与三维世界联系起来。
1. 已有多种技术尝试将二维检测扩展到三维，但这些技术都需要场景的先验知识，且不能很好地推广。
2. 随着 3D 传感器 (如 Microsoft Kinect) 的出现，提供深度和颜色信息，将 2D 知识传播到 3D 的任务变得更加容易实现。三维目标检测的重要性在于提供更好的定位，将知识从图像帧扩展到现实世界。这使得机器 (例如机器人) 与其环境之间能够进行交互。由于三维检测的重要性，许多技术都利用大规模的 RGB-D 数据集，尤其是 SUN RGB-D，将二维边界框替换为三维边界框，它为数百个目标类提供了三维边界框注释。

最先进的 3D 检测方法的一个缺点是运行时。尽管硬件加速 (GPU)，它们往往比 2D 目标检测方法慢得多，有几个原因。(i) 其中一个原因是三维场景相对于二维图像场景的相对大小。添加额外的空间维度会大大增加 3D 中的搜索空间，从而降低搜索速度。(ii)另一个原因是单个 RGB-D 图像生成的三维点云中可用的稀疏数据不完整，存在二维图像中存在的弱邻接 / 邻近特征。(iii) RGB-D 图像中深度信息的理想编码和挖掘仍然是一个开放的挑战。

文献中的技术要么尝试增加颜色通道的深度，要么将其编码为稀疏体素化的三维场景。使用深度信息作为额外的通道有助于检测过程，同时仍然受益于快速的 2D 操作，但最终结果仅限于 2D 检测，其形式为 2D 边界框或 2D 对象分段。可以用 3D 编码的信息包括密度、法线、梯度、符号距离函数等。然而，所有这些基于三维体素的技术都存在大量的三维信息缺失，使得场景中的可观测点只构成了三维体素的一小部分。Information that can be encoded in 3D include densities, normals, gradients, signed distance functions, among others. Nonetheless, all these 3D voxelization-based techniques suffer from the large amount of missing 3D information, whereby the observable points in a scene only constitute a small fraction of 3D voxels.

在本文中，我们提出了一种三维物体检测方法，它得益于二维物体检测的进展，可以快速检测三维边界框。我们的方法的输出如图1所示。

![image-20210625122139059](https://oj84-1259326782.cos.ap-chengdu.myqcloud.com/uPic/2021/06_25_image-20210625122139059.png)

我们没有改变二维技术来接受三维数据，因为三维数据可能是缺失的或没有得到很好的定义，而是利用二维技术来限制我们三维检测的搜索空间。然后，我们利用三维信息来确定所需物体的方向、位置，并对其周围的边界框进行评分。我们使用以前的方法来独立确定每个物体的方向，然后使用获得的旋转和每个方向的点密度来回归物体的边缘。我们最终的三维包围盒得分是利用语义背景信息重新确定的。

除了通过对可能包含特定物体的三维场景的部分进行磨练而获得的速度外，三**维搜索空间的减少也有利于检测器的整体性能。这种减少使得三维搜索空间比从头开始搜索整个场景更适合于三维方法，后者会减慢搜索速度并产生许多不良的假阳性。**这些误报可能会混淆三维分类器，而三维分类器比二维分类器更弱，因为它是在稀疏的（大部分是空的）三维图像数据上训练的。

2、相关工作
======

在计算机视觉技术方面有丰富的文献，通过在物体周围放置矩形框来检测物体。我们这里提到的一些最具代表性的方法, 解决这一问题, 即 DPM(可变形部件模型) 和选择性搜索在无处不在的深度学习的基础方法, 以及代表深度网络这个任务包括 R-CNN,Fast R-CNN,Faster R-CNN, ResNet, YOLO,R-FCN。所有这些技术都只在二维图像平面上对目标进行检测，并且已经发展到非常快速和高效的程度。随着 3D 传感器的出现，已经有许多工作使用 3D 信息来更好地定位目标。在这里，我们提到了几种方法，它们研究在深度信息存在下的目标检测。其他基于 RGB 和深度的语义分割图像的技术。所有这些 3D 感知技术都使用额外的深度信息来更好地理解二维图像，但并不旨在将正确的 3D 包围框放置在被检测目标周围。

1. [30]方法利用多视点三维 CAD 模型的效果图对整个空间滑动窗口得到的所有三维边界框进行分类。使用 CAD 模型限制了可以检测到的类和目标的多样性，因为找到不同类型和目标类的 3D 模型要比拍摄它们困难得多。
2. 此外，滑动窗口策略的计算要求很高，使得这种技术相当慢。类似的检测器使用目标分割和姿态估计来表示在编译库中具有相应 3D 模型的目标。

尽管如此，我们相信正确的 3D 边界框布局有利于这样的任务，并且可以根据这些模型的可用性来执行模型拟合。与 [7] 相比，我们的方法不需要三维 CAD 模型，对二维检测误差不敏感，利用上下文信息提高了检测效率。其他的方法提出了 3D 盒子，并根据手工制作的功能进行评分。

1. 在 [1] 中提出的方法在自动驾驶环境下，将三维包围框放置在物体周围。该问题在 MRF 中被表述为推理，MRF 生成 3D 提案，并根据手工制作的功能对提案进行评分。该方法使用立体图像作为输入，只针对少数特定于街景的类。
2. 对于室内场景，[19]中提出的方法是利用二维分割提出候选框，然后通过形成一个条件随机场 (CRF) 将不同来源的信息进行分类。
3. 最近的 [26] 方法提出了一种面向梯度描述符的云，并将其与法线和密度一起用于对三维边界框进行分类。该方法还利用上下文特征，利用 [11] 级联分类框架，更好地提出三维框。该方法在 SUN-RGBD 上取得了最先进的性能; 然而，计算所有三维长方体假设的特征非常慢(每节课 10-20 分钟)。

贡献：

1. 提高速度：我们提出了一种快速的技术，只使用 RGB-D 数据在目标周围放置包围框。我们的方法不使用 CAD 模型，而是放置 3D 包围框，这使得它很容易推广到其他目标类。
2. 避免假阳性，加速：通过仔细研究特定目标实例在 3D 中的位置 (使用 2D 检测)，我们的 3D 检测器不需要对整个 3D 场景进行彻底的搜索，并且遇到的假阳性可能会更少，从而使其混淆。与直接在三维中工作的两种最先进的三维探测器相比，我们的方法在不牺牲检测精度的前提下实现了加速。

3、方法
====

给定一个 RGB 图像及其对应的深度图像，我们的目标是在一个已知类的目标周围放置 3D 包围框。我们的 3D 目标检测管道由四个模块组成 (查看图 2)。

1. 在第一个模块中，我们使用了最先进的 2D 目标检测方法，特别是 Faster R-CNN，来定位可能的目标周围的 2D 边界框。每个 2D 边界框在 3D 中扩展到我们所说的截锥体。（2D信息连续、更快、固定区域搜索）
2. 在第二个模块中，不像之前的方法[31] 假设场景中的所有目标都具有相同的方向，我们估计场景和单个目标的方向，其中每个目标都有自己的方向。
3. 在第三个模块中，我们训练了一个多层感知器，利用方向上的点密度，在每个方向上回归三维物体的边界。
4. 在最后一个模块中，我们使用基于目标类再现和类间距离的上下文信息来细化检测分数。

                        ![](https://img-blog.csdnimg.cn/20190925230014840.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNjY3MDUyOQ==,size_16,color_FFFFFF,t_70)

3.1、2D检测
--------

我们的 3D 检测管道的第一步是在 2D 中获得目标位置的初始估计。在这里，我们选择使用 Faster R-CNN 模型和 VGG-16 net 对 3D dataset (SUN-RGBD)中的一组目标类 (object classes) 进行检测器的训练。在 2D 中，检测到的目标由 2D 窗口表示。在 3D 中，这转化为一个 3D 扩展，我们称之为截锥体。物体的截锥体对应于在二维检测窗口中包含投影到图像平面上的三维点。这样，在二维图像中，通过相机中心的平面和边界框的线段所限定的区域内就会出现潜在的物体。截锥体类似圆锥，底部为长方形。从本质上讲，这个区域提供的三维搜索空间要比 RGB-D 传感器捕捉到的整个区域小得多。此外，每个 frustum 只指定 2D 检测器返回的目标类。

二维目标检测得益于图像中信息的连续性，二维卷积包含了所有目标位置的 RGB 信息。相对于三维场景体素化中缺失的三维数据，这使得二维信息对于目标检测和分类更加可靠。此外，通过 Faster 的 R-CNN 的转发在 GPU 上至少运行 5 帧每秒，这比使用 3D 卷积的深度滑动形状 (DSS) 快两个数量级。在每个截锥体中，三维点分布在深度最小的点和深度最大的点之间。这些点保留了正确检测目标所需的所有深度信息。与穷举滑动窗口方法相比，这类似于固定在特定的 2D 区域，而不是搜索整个区域来查找所有目标类。

3.2、估计三维方位
------------

到目前为止，我们已经确定了最有可能包含目标类的区域。我们的下一步是估计这个区域内物体的方位。由于三维包围框是曼哈顿结构，因此目标的方向必须与最佳曼哈顿框架对齐。这个框架可以估计物体的方向，因为在室内场景中发现的大多数三维物体都可以近似为曼哈顿物体，其法线与三个主要正交方向对齐。为了计算这个曼哈顿帧，我们使用了在 [4] 中提出的曼哈顿帧估计 (MFE) 技术来独立估计物体在每个截锥内的方向。综上所述，旋转可以通过求解以下优化问题得到

                                  ![](https://img-blog.csdnimg.cn/20190925225642828.png)

其中 N 是矩阵包含每一个 3D 点的法线,λ是一个常数参数, X 是一个松弛变量引入 RN 稀疏。这里，我们假设每个截锥体中只有一个主要对象。我们首先计算图像中所有 3D 点的法线，并使用 MFE 对整个场景的摄像机进行定向。对于每个截锥体，我们使用房间方向初始化，并使用其中点的法线来估计目标的方向。在本文中，我们修改了 MFE，以限制旋转是围绕轴沿地板法线 (偏航角只)。这个限制对于室内场景中的大多数目标都是一个可行的假设，并且与 SUN RGB-D dataset 的注释方式一致。对于非曼哈顿结构的物体 (如圆形物体)，许多方向被认为是正确的。MFE 技术的输出仍然是一种可行的目标检测方向。

3.3、Bounding Box 回归
-------------------

在此步骤中，我们需要匹配最能描述被检测对象的 3D 包围框。给出了在截锥体内的三维点以及物体的估计方向，并以三维点的质心为中心，用估计方向建立了一个标准正交系统。然后，我们为沿每个方向的三维点的坐标构造直方图。然后，这些直方图被用作多层感知器 (MLP) 网络的输入，MLP 网络学习从训练数据中返回目标边界框的边界。对于每个目标类，训练一个具有隐含层的网络，以坐标直方图作为输入，输出目标沿各个方向的边界框边界。直方图描述了每个方向上点的密度，而高密度对应于表面位置。我们选择一个固定的箱子大小来保持真实的距离，并选择一个固定的直方图长度来考虑所有的物体大小。补充材料中给出了 MLP 的输入示例。每个方向的训练都是分开进行的，即长、宽、高。在测试过程中，高度由地面方向确定，长度和宽度由截锥体内各方向点的较宽分布确定。为了形成训练集，我们使用了 2D groundtruth 窗口和 groundtruth 3D 框。由于许多室内物体都放置在地板上，我们使用训练集中的高度信息来剪辑接近地板的物体的高度，从而从地板开始。一旦得到三维边界框，我们就给它分配一个分数，这个分数是两个分数的线性组合:(1)初始二维检测分数，(2)三维点密度分数。通过对所有类的三维长方体的三维点云密度训练线性 SVM 分类器，得到三维点云密度分数。这个简单的 3D 功能类似于 [26] 中使用的功能，并受到了它的启发。显然，其他 3D 功能也可以被纳入，但要付出额外的计算成本。我们使用所有可能的目标旋转，以及对象位置的细微变化来训练分类器。

3.4、基于上下文信息的细化
--------------

给定一组 3D 边界框![](https://private.codecogs.com/gif.latex?%5C%7B%20B_i%20%3A%20i%20%5Cin%20%5C%7B%201%2C2%2C...%2Cn_b%5C%7D%5C%7D)其中![](https://private.codecogs.com/gif.latex?n_b)为 box 的数量，我们的目标是根据上下文信息来细化它们的分数。我们将一个标签 li 关联到每个 box![](https://private.codecogs.com/gif.latex?B_i)，其中![](https://private.codecogs.com/gif.latex?l_i%20%5Cin%20%5C%7B%200%2C1%2C...%2Cn_l%20%5C%7D)。这里![](https://private.codecogs.com/gif.latex?n_l)是被考虑的对象类标签的数量，而 zero 标签对应于背景。我们假设边界框标签![](https://private.codecogs.com/gif.latex?L%20%3D%20%5C%7Bl_1%2C%20l_2%20%2C...%2C%20l_n%20%5C%7D)是一组离散随机变量有一个相关联的吉布斯分布因子图 g 因子图是由一组变量节点 (边界框标签), 和一组节点 P 的因素, 我们选择的任意组合 2 边界框。该图是通过将给定场景中的所有目标分配给一个特定节点来构造的。该图是通过将给定场景中的所有对象分配给一个特定节点来构造的。我们的图模型如图 3 所示。在这种情况下，我们有

                               ![](https://img-blog.csdnimg.cn/20190927211634340.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNjY3MDUyOQ==,size_16,color_FFFFFF,t_70)

                                           ![](https://img-blog.csdnimg.cn/20190927211659970.png)

这里 U 和 B 是一元和二元对数势函数。我们的目标是找到最大化 aposteriori (MAP) 的标签，

                                                  ![](https://img-blog.csdnimg.cn/20190927211756585.png)

通过引入局部边缘变量![](https://private.codecogs.com/gif.latex?p_u)和![](https://private.codecogs.com/gif.latex?p_b)，将上述问题转化为线性规划 (LP)，

                                      ![](https://img-blog.csdnimg.cn/2019092721193093.png)

**与一元项有关的概率：**摘要为了对一元势 pu 模型进行建模，利用三次多项式核的一对一 SVM 分类器，训练了一个纠错输出码多类模型。我们增加了两种类型的特征，几何特征和深度学习特征。几何特征包括长度、宽度、高度、纵横比和体积。为了提取深度学习特征，我们将 3D 框的重投影到图像平面上，运行 Fast RCNN，利用全连通层 (FC7) 的特征。在测试过程中，我们将分类分数转换为后验概率，并将其作为一元概率使用。

**与二进制项有关的概率：**对于每一对 3D 盒子，我们计算一个盒子被分配一个标签![](https://private.codecogs.com/gif.latex?l_i)的概率，前提是另一个标签是![](https://private.codecogs.com/gif.latex?l_j)。我们利用了发生在三维场景中的两个关系，类的共现和类的空间分布。对于共发生概率![](https://private.codecogs.com/gif.latex?p_o)，我们使用训练集中两个类的每个组合的共发生次数。给定标签![](https://private.codecogs.com/gif.latex?l_i), ![](https://private.codecogs.com/gif.latex?p_o)为![](https://private.codecogs.com/gif.latex?l_j)的发生次数与其他所有发生次数的比值。对于空间关系，我们使用核密度估计 (KDE) 来分配基于一对边界盒之间 Hausdorff 距离的概率![](https://private.codecogs.com/gif.latex?p_d)。最后, 我们最后的二元词概率定义为:![](https://img-blog.csdnimg.cn/20190927212549744.png)。

为了在一元项和二元项之间进行权衡，我们使用 softmax 操作符。为了推断出最终的标签集，我们使用了 [22] 的 LP-MAP 技术。然后，我们将最后一组标签与最初的标签进行比较，并增加保留最初标签的标签的得分。

4、实验
====

**我们在 SUN RGB-D 数据集上评估了我们的技术，并与两种最先进的方法: 深层滑动形状 (DSS) 和定向梯度云 (COG) 进行了比较。**我们采用组委会选择的 10 个类别进行训练和测试。我们还使用了 [31] 中提供的数据集修改，它提供了楼层方向。我们采用与这两种方法相同的评价尺度，即假设所有的边界框都与重力方向对齐。

**评估准则：**我们的评估是基于传统的三维体积交联 (IoU) 措施。如果使用 groundtruth 框生成的边界框的卷 IoU 大于 0.25，我们认为检测是正确的。我们遵循 DSS 和 COG 采用的相同评估策略。我们绘制了 10 个类的精确回忆图，并计算了曲线下的面积，用平均精度 (AP) 表示。与前面的方法类似，我们将其与 amodal groundtruth 包围框进行比较，后者扩展到可见范围之外。

**实验步骤：**在我们的 2D 目标检测器中，我们遵循数据集增强约定，将翻转后的图像添加到训练集中。我们使用在 ImageNet 上预先训练的模型，初始化 Fast R-CNN 网络中的所有卷积层。我们还采用了 Faster R-CNN 中描述的 4 步交替训练。在我们所有的实验中，我们考虑沿垂直于地面方向的所有三维点坐标的第一个百分位作为相机高度。在我们最初的 2D 检测之后，我们删除所有重叠超过 30% 的得分较高的框，并删除所有得分非常低的框 (小于 0.05)。为了确定主曼哈顿框架的方向，我们对法线进行了 10 次子采样，这使得对管道的这一阶段进行实时处理成为可能。一旦估计了房间的朝向，我们就并行地计算每个截锥体的朝向和回归。关于边界框回归的实现(第 3.3 节) 和使用上下文关系的细化 (第 3.4 节) 的进一步细节载于补充材料。

**计算速度：**我们在 MATLAB 中实现了我们的算法，使用 12 核 3.3 GHz CPU 和 Titan X GPU。Faster R-CNN 训练和测试是用 Caffe 完成的。初始目标检测仅需 0.2 秒。当并行运行时，所有目标的方向估计大约需要 1 秒。对于沿 x、y 和 z 方向的所有目标，MLP 回归大约需要 1.5 秒。基于上下文的对象细化运行时间为 1.45 秒，包括设置因子图和推断最终标签的时间。综上所述，我们的检测方法对每一对 RGB-D 图像总共需要 4.15 秒。

4.1、定量比较
--------

首先，我们研究了拟议管道中每个阶段的重要性。我们为方法的不同变体绘制了精确回忆曲线 (参见图 4)。在本例中，我们使用 2D 目标检测器输出的标签及其相应的分数。(2) 我们还尝试将盒子与 frustums 匹配，即所有 frustums 中的所有盒子都具有相同的朝向 (房间朝向)。(3) 我们算法的最后一个变体没有使用 MLP 回归器回归目标边界。我们将回归框替换为一个向各个方向扩展到最大值和最小值坐标的百分位数的框。显然，这不能处理 amodal 框。

          ![](https://img-blog.csdnimg.cn/20190927213737735.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNjY3MDUyOQ==,size_16,color_FFFFFF,t_70)

**1. 固定方向 vs 独立方向：**我们研究了正确定位三维边界框的重要性。相对于一个固定的方向，为每个边界框计算正确的方向会增加最终的得分 (表 1)，这是因为相同方向的目标之间有更高的重叠，而且方向对于在 MLP 回归器中匹配正确的目标边界至关重要。

**2. 回归的重要性：**回归的重要性在于定位目标中心和估计边界盒维数。由于三维数据的噪声性质以及背景点的存在，使得三维点在截锥体内的质心与物体的质心不同。如果不进行回归，检测分值会显著下降 (表 1)，这是由于与检测分值直接相关的目标大小不正确造成的。此外，由于 groundtruth 框是 amodal 的，这意味着它们扩展到可见部分之外，因此需要回归来扩展超出可见点之外的框。

                      ![](https://img-blog.csdnimg.cn/20190927214144441.png)

**3. 纹理信息的重要性：**在我们的技术的最终形式中，我们结合上下文信息来细化最终检测。这种细化将 mAP 得分提高 2% 以上 (表 1)。与图 4 中的“no refinement”” 变体相比，我们注意到细化在相同的召回率下提高了精度。另一方面，它实现了相同的最大召回，因为它不改变 3D 框的数量及其位置。在第二部分，我们比较了两种最先进的方法，DSS 和 COG(表 1)，我们直接从 [31] 和[26]报告运行时和 AP 结果。我们提供了我们的技术的两个版本: 最终版本 (包括管道中的所有模块) 和非细化版本 (不使用上下文信息，但是大约快 35%)。非细化版本的速度比 DSS 快 7 倍，在精度方面稍好一些。最终版本比 DSS (45.1% mAP vs . 42.1% mAP) 更快 4.7×，准确率更高 3%。它也比 COG 大约快两个数量级，同时仍然实现了类似的检测性能。

4.2、定性比较
--------

现在，我们展示一些定性的结果。在图 5 中，我们将方法的检测结果 (红色部分) 叠加到 SUN-RGBD 的 8 张 RGB-D 图像的三维点云上，并将其与 3D groundtruth boundingbox(如图 ingreen 所示)进行比较。我们的方法能够根据方向和范围正确地放置边界框。我们还在图 6 中显示了我们所提议的技术的错误检测。这包括在 2D 中没有检测到的目标，或者使用 MLP 的输出将目标放错位置的对象。由于背景点或不正确的回归而发生错位。此外，假阳性还包括来自不可见对象类的目标，这些目标看起来类似于在训练中看到的类(例如。柜台 vs. 书桌，抽屉 vs. 梳妆台)。

          ![](https://img-blog.csdnimg.cn/2019092721543972.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNjY3MDUyOQ==,size_16,color_FFFFFF,t_70)

           ![](https://img-blog.csdnimg.cn/20190927215452297.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl8zNjY3MDUyOQ==,size_16,color_FFFFFF,t_70)

5、结论
====

提出了一种快速的室内场景三维目标检测算法。我们利用二维检测技术对三维中特定对象类的潜在三维位置进行了挖掘，从而实现了简单的三维分类器和搜索机制。与目前两种最先进的方法相比，我们的方法比其中一种方法快 3 倍，实现了更好的(+3% mAP) 检测性能，比另一种方法快两个数量级，同时仍然可以比较具有挑战性的大型 SUN-RGBD 数据集。

