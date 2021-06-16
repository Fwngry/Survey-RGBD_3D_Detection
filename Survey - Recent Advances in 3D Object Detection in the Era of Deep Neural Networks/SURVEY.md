# Survey

摘要

随着深度学习技术和其他强大工具的快速发展，三维物体检测取得了巨大的进步，成为计算机视觉中发展最快的领域之一。许多自动化应用，如机器人导航、自动驾驶、虚拟或增强现实系统，都需要估计准确的三维物体位置和检测。在这种要求下，许多方法被提出来以提高三维物体定位和检测的性能。尽管最近做出了努力，但由于遮挡、视角变化、尺度变化和三维场景中的有限信息，三维物体检测仍然是一项非常具有挑战性的任务。**在本文中，我们对三维物体检测技术中最近的最先进方法进行了全面的回顾。**

我们从一些基本概念开始，然后描述一些可用的数据集，这些数据集旨在促进三维物体检测算法的性能评估。

接下来，我们将回顾该领域最先进的技术，强调其贡献、重要性和局限性，作为未来研究的指南。

最后，我们对最先进的方法在流行的公共数据集上的结果进行了定量比较。

[TOC]

物体检测是计算机视觉中的基本问题之一。最近，深度神经网络的迅速成功极大地推动了各种基于自动化的系统的发展，如移动机器人、自动驾驶和虚拟或增强现实系统，这些系统急切地需要3D理解。三维物体检测有助于提取三维空间中物理物体的几何理解。在过去的几年里，尽管在基于图像的二维物体检测[1]-[3]和实例分割[4]任务中取得了很大的进展，但文献中对三维物体检测的探讨却很少。

除了二维检测，由于物体之间复杂的相互作用、严重的遮挡、杂乱的场景、视角和比例的变化以及三维数据提供的有限信息，在三维世界中准确的三维理解仍然是一个公开的挑战。此外，三维数据本身的表示更加复杂，由于增加了一个额外的维度，需要更高的计算和内存要求。

最近3D传感设备（如LiDAR、Radar）传感器的进步和低成本设备（如Microsoft Kinect、Xtion Pro-live等）的出现，使得3D数据的采集比以往任何时候都更加方便。利用3D传感技术的力量和快速发展的深度学习技术，3D物体检测研究引发了解决挑战的新兴趣，使其成为计算机视觉中不断变化的研究领域。这种快速增长使得跟踪处理三维物体检测技术的方法变得非常困难。

我们的调查重点是描述和分析最近竞争激烈的基于深度学习的三维物体检测方法。关于3D数据[5]和2.5/3D室内场景理解[6]的调查为数不多，只涉及特定领域的现有方法，由于3D物体检测是一个快速增长的领域，它可能缺乏最先进的方法的想法，可以提供一些新的解决方案和方向。在本文中，我们系统而全面地回顾了最新的三维物体检测方法，并确定了这些方法的潜在优点和缺点。我们列出了最近提出的解决方案，但忽略了对传统方法的讨论，因此读者可以更容易地看到三维物体检测的前沿技术。根据输入模式，我们将现有的方法分为三类：基于图像的处理方法、基于点云的处理方法和基于多模态融合的方法。最后，我们提供了3D物体检测方法的比较，并讨论了当前的研究问题和未来的研究方向。因此，我们相信我们的工作将被作为一个有用的参考，并对研究界做出重大贡献。

我们工作的主要贡献有以下几点。

- 我们首先介绍了三维边界框编码技术、三维数据表示和传感模式的基本概念，然后介绍了一些现有的数据集，这些数据集可能对未来基于深度学习的三维物体检测项目有帮助。
- 我们对最近的三维物体检测方法、它们的起源、贡献和局限性进行了深入系统的回顾。
- 我们在流行的数据集上对所描述的方法进行了比较，并展示了它们的性能。
- 我们指出了潜在的研究挑战、差距和未来的研究方向。





新场景：例如移动机器人，自动驾驶以及迫切需要3D理解的虚拟或增强现实系统。 3D对象检测有助于提取对3D空间中物理对象的几何理解。

研究空间：尽管在基于图像的2D对象检测[1] – [3]和实例分割[4]任务方面已经取得了很大的进步，但是在文献中很少探索3D对象检测。

存在挑战（问题定义）对象之间的复杂相互作用，严重的遮挡，混乱的场景，视点和比例变化以及3D数据提供的信息有限，对3D世界的准确3D理解仍然是一个开放的挑战。（技术难题）此外，3D数据本身的表示更加复杂，由于添加了额外的维度，因此需要更高的计算和内存要求。

## <3D bonding box编码方式>

3D bonding box：3D空间中的有向立方体
3种方法被用于表示3D bonding box：

1. axis aligned 3D center offset method [7], 

2. 8-corners method [8] and

3. 4-corner-2-height method [9]

   ![边界框编码方式](https://oj84-1259326782.cos.ap-chengdu.myqcloud.com/uPic/2021/03_25_%E8%BE%B9%E7%95%8C%E6%A1%86%E7%BC%96%E7%A0%81%E6%96%B9%E5%BC%8F.png)

   3D center offset method 3D中心偏移方法[7]将3D边界框参数化为（∆x，∆y，∆z，∆h，∆w，∆l，∆θ）， 
   其中（∆x，∆y，∆z）是3D边界框的中心坐标，
   （∆h，∆w，∆l）分别是框的高度，宽度和长度，
   而∆θ是偏航角yaw angle 边界框的角度。
   俯仰角和侧倾角被认为是零，或者对于该任务而言重要性不大。The pitch and roll angles are considered to be zero, or to be of less importance for this task. 

   在8-corners方法[8]中，将3D边界框的参数设置为（∆x 0。。。∆x 7，∆y 0 ...Δy7，Δz0…Δz7，其中每个（x，y，z）代表3D边界框的每个角。

   4-corner-2-height方法[9]对3D边界框进行编码，该边界框具有4个角和2个高度值，这些值表示相对于地平面的顶部和底部角偏移，并定义为（∆x 1。。∆x 4， Δy1…Δy4，Δh1，Δh2）

## <3D数据表示形式>

3D数据可以具有多个数据表示形式：例如体素表示形式[11]，点云[12]，多视图图像[8]，深度通道编码[13]，多边形网格，截断符号距离函数（TSDF）[7] ，构造实体几何（CSG），圆图[14]，八叉树[15]和基于图元的表示。 数据是通过各种传感器捕获的，这些传感器现在已成为许多汽车，无人机，机器人和智能手机的标准功能。

1. 单目摄像机
2. LiDAR
3. RGB-D
4. Radars
5. Ultrasonics

## <数据集>

![DataSets](https://oj84-1259326782.cos.ap-chengdu.myqcloud.com/uPic/2021/03_25_03_25_DataSets.png)

## <基于输入数据对3D目标检测方法分类>

3D目标检测系统将不同类型的数据作为输入，并输出3D边界框和传感器视场中所有关注目标的类别标签。
输入数据可能来自各种传感器的组合，例如单眼或立体摄像机，RGB-D传感器，LiDAR和声纳。 现有的大多数作品要么仅使用视觉摄像机的RGB图像，要么仅使用LiDAR的3D点云，或者将RGB图像与3D LiDAR点云结合起来，或者使用Kinect RGB-D摄像机融合RGB图像和深度图像。

基于输入数据表示，3D目标检测方法可分为三大类：
1. 基于图像的方法
2. 基于点云的处理方法
3. 基于多峰融合的方法

## <基于图像的方法>

1. 仅将单目图像用作输入
2. 由于没有可用的深度信息，因此大多数方法都使用两阶段方法进行3d目标检测：首先生成 候选，然后执行检测。
从<手工制作特征> OR 基于深度学习的2D目标检测提取得到2D 候选框；然后使用<几何约束>、<3D模型拟合>或<活动形状建模>将其回归成带方向的3D bounding box

Active shape modeling （1）（2）（3）

3D models ﬁtting（4）（5） 多数据集进行恢复（6）

geometrical constraints 几何特性+网络调整（7）（8）

### 1. Hand-Crafted Feature 手工提取特征

> [46]  C. L. Zitnick and P. Dolla ́r, “Edge boxes: Locating object proposals from edges,” in *European Conference on Computer Vision (ECCV)*, 2014, pp. 391–405.
>
> [47]  P.Kra ̈henbu ̈hlandV.Koltun,“Learningtoproposeobjects,”in*IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2015, pp. 1574–1582.
>
> [48]  T. S. H. Lee, S. Fidler, and S. J. Dickinson, “Learning to combine mid- level cues for object proposal generation,” in *2015 IEEE International Conference on Computer Vision (ICCV)*, 2015, pp. 1680–1688.

### 2. Active Shape Modeling 活动建模

Active Shape model 是物体形状的统计模型，意图将model在新图片中得到匹配。通过迭代变形以适应新图像中的样本。物体形状是由PDM（Point  Distribution Model）构建并通过训练集中的标签数据集来变形，这也是唯一的一种途径。

1. 为图中每个点寻找合适的位置以生成“suggested shape”，这个过程中用到的是“profile model”，寻找有力的边界或利用Mahalanobis distance使model template与点得以匹配。
2. 将“suggested shape”与“Point  Distribution Model”相适应
3. 

> [55]  M. Z. Zia, M. Stark, and K. Schindler, “Explicit occlusion modeling for 3d object class representations,” in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2013, pp. 3326–3333.
>
> [56]  Y. Lin, V. I. Morariu, W. H. Hsu, and L. S. Davis, “Jointly optimizing 3d model fitting and fine-grained classification,” in *European Confer- ence on Computer Vision (ECCV)*, 2014, pp. 466–480.

### （1）3DOP[57]

> 为了得到高质量3D目标proposals，将立体图像作为输入，对双目立体图像编码所得的能量函数最小化解决问题，化为马尔可夫随机场中的编码对象大小先验。计算深度，使用深度来计算点云x，并在此域中进行直接在3D中利用立体图像和推理，地平面和各种深度信息特征的推理。

> X. Chen, K. Kundu, Y. Zhu, A. G. Berneshawi, H. Ma, S. Fidler, and R. Urtasun, “3d object proposals for accurate object class detection,” in *Proceedings of the 28th International Conference on Neural Infor- mation Processing Systems (NIPS)*, 2015, pp. 424–432.

利用立体图像重建深度 stereo imagery to reconstructs depth，并利用马尔可夫随机场（MRF）排序模型以下生成 3D边框形式的目标候选。

将这些3D候选作为2D边界框投影到图像上，并提供给扩展的Fast R-CNN [10]Pipeline共同预测目标 候选的类别，通过使用角度回归angle regression来估计目标方向。
3DOP以能量最小化方法，通过编码目标尺寸先验信息，地平面和各种深度特性（例如自由空间，盒子内的点密度，可见性和与地面的距离）。
尽管作者展示了3DOP在2D检测和方向估计方面的出色性能，但他们并未对3D边界框 候选进行定量评估。

### （2）Mono3D[58]

> 单目版本，评分系统+引入假设
>
> X. Chen, K. Kundu, Z. Zhang, H. Ma, S. Fidler, and R. Urtasun, “Monocular 3d object detection for autonomous driving,” in *IEEE* 209.Conference on Computer Vision and Pattern Recognition (CVPR),2016, pp. 2147–2156.

Mono3D扩展了3DOP ，以获得单目版本。

评分系统：Mono3D从单目图像生成3D候选，并使用所有目标应靠近地平面的假设，该地平面应与像平面正交。 通过几个直观印象（利用语义分割，上下文信息，大小和位置先验以及典型的目标形状）在图像平面中对每个3D 候选框进行评分。

最后，将最高得分的区域送到3DOP中使用的extended Fast R-CNN pipeline并预测类别标签并估计3D边界框的偏移量和方向。 

但是，这些模型的局限性在于它需要为每个目标类别单独运行，并且需要很多候选才能实现较高的召回率，这导致分类处理时间增加。

### （3）DeepStereoOP [59]

> C. C. Pham and J. W. Jeon, “Robust object proposals re-ranking for object detection in autonomous driving using convolutional neural networks,” *Sig. Proc.: Image Comm.*, vol. 53, pp. 110–122, 2017.

为了克服该限制，提出了DeepStereoOP [59]，一种独立于类的、目标候选重新排序方法，该方法在轻量级两流CNN架构中同时使用单眼图像和深度图，优于3DOP和Mono3D方法。

### 3. 3D Models Fitting 3D模型适应

> [52]  B. Pepik, M. Stark, P. V. Gehler, T. Ritschel, and B. Schiele, “3d object class detection in the wild,” in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2015, pp. 1–10.
>
> [53]  M. Aubry, D. Maturana, A. A. Efros, B. C. Russell, and J. Sivic, “Seeing 3d chairs: Exemplar part-based 2d-3d alignment using a large dataset of CAD models,” in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2014, pp. 3762–3769.
>
> [54]  J. J. Lim, A. Khosla, and A. Torralba, “FPM: fine pose parts-based model with 3d CAD models,” in *European Conference on Computer Vision (ECCV)*, 2014, pp. 478–493.

### （4）3DVP[11]

> 3D模型适应
>
> Y. Xiang, W. Choi, Y. Lin, and S. Savarese, “Data-driven 3d voxel patterns for object category recognition,” in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2015, pp. 1903– 1911.

通过学习这么大的3DVP字典，可以有效地模拟由于类内变异和遮挡而导致图像中目标亮度的变化，其中每个3DVP都捕获了上面列出的三个属性的特异性。 （外观，3D形状和遮挡）
3DVP训练检测器
即使从任何角度观察或在严重遮挡下仍可见时，这些检测器都可以对图像中的目标进行定位。 在测试阶段，3DVP检测器可以预测2D分割蒙版，3D姿势或3D形状，并且可以与3D CAD模型一起使用以检测3D目标。

### （5）SubCNN[51]

> Based 3DVP，利用到“区域候选网络”的好性能
>
> Y. Xiang, W. Choi, Y. Lin, and S. Savarese, “Subcategory-aware convolutional neural networks for object proposals and detection,” in *IEEE Winter Conference on Applications of Computer Vision (WACV)*, 2017, pp. 924–933.

在基于CNN的对象检测中，区域提议网络（RPN）比传统的区域提议方法表现更好。 瓶颈在于它无法处理对象的比例变化，遮挡和截断。

1. 区域候选网络：始发于Fast R-CNN
    改进：使用“子类别卷积层subcategory convolutional layer”，输入“图像金字塔image pyramids”，在多尺度上计算卷积特征，以进行有大变化的目标检测，输出特定子类别在特定位置或范围下的存在 - - heat map
2. 子类别 SubCategory：3DVPs（By using 3DVPs as subcategories, the method can jointly detect objects, 3D shape, pose, and occluded or truncated regions.）
    3D体素表示可用于解决遮挡问题；但待检测目标的姿态与现有模式不相同时，可能会失效。

### （6）DeepMANTA（Vehicles）[50]

> 利用2个数据集对目标进行恢复、匹配、评分；解决遮挡、截断、自遮挡问题
>
> F. Chabot, M. Chaouch, J. Rabarisoa, C. Teulie`re, and T. Chateau, “Deep MANTA: A coarse-to-fine many-task network for joint 2d and 3d vehicle analysis from monocular image,” in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 1827– 1836.

即使发生了遮挡、截断、自遮挡 occlusion, truncation or selfocclusion

Deep MANTA使用多任务CNN共同生成具有多个优化步骤的车辆2D和3D边界框。

1. 输入单目图像到多目标CNN中，输出2D scored bounding boxes, vehicle part coordinates, 3D template similarity, and part visibility properties.

2. 利用两个3D数据集（包含3DBBox的坐标和尺度）来恢复交通工具Object的位置和方向

3. 对比得到数据集匹配最佳的实体，通过算法进行评分。

相较于3DOP准确度提高；但依赖于大数据集。

### 4. Geometrical Constraints 几何约束

### （7）Deep3DBox[49]

> 结合几何特性的深度神经网络，deep neural networks combining with geometric properties
>
> A. Mousavian, D. Anguelov, J. Flynn, and J. Kosecka, “3d bounding box estimation using deep learning and geometry,” in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 5632–5640.

在Deep3DBox [49]中提出了一种非常简化的单眼仅图像结构，该结构将视觉外观和几何约束组合到3D目标检测场景中。 Deep3DBox利用最先进的2D目标检测器来估算3D边界框（其几何约束条件是3D边界框必须紧贴2D检测窗口，要求2D边界框的每一侧都必须通过3D框角中的至少一个）

（MultiBin回归）首先，通过训练离散的连续CNN架构来扩展2D目标检测器[61]，以回归目标3D边界框的方向和尺寸。与仅使目标的3D方向回归相反，Deep3DBox使用MultiBin回归来估计目标的航向角。

最后，根据估算的目标的航向角，尺寸和上述约束，可以使用基于优化的方法来计算中心坐标以及完整的3D边界框。与更复杂的3DOP架构相比，Deep3DBox已显示出改进的检测和方向估计性能[57]。

### （8）Multi-level fusion [62]

> B. Xu and Z. Chen, “Multi-level fusion based 3d object detection from monocular images,” in *IEEE Conference on Computer Vision and**Pattern Recognition, (CVPR)*, 2018, pp. 2345–2353.

继Deep3DBox的 MultiBin体系结构 之后，提出了一种端到端多级融合的框架。

该框架对单个单目图像进行2D / 3D目标检测，并包含了基于独立全卷积网络（FCN）的模块，来预测视差信息并计算3D点云 

然后将视差信息用“正视图特征表示 front view feature representation” 进行编码，并与原始RGB图像融合以增强输入，将其输入基于Faster R-CNN的区域候选网络，生成2D区域候选。

基于2D区域候选，将RoI最大池化层应用于主卷积分支；RoI均值池层将候选内的点云转换为另一流中的固定长度特征向量。 然后使用不同级别的融合来计算目标的分类，方向，尺寸和位置 classiﬁcation, orientation, dimension, and location。

### 5. Pseudo-LiDAR Representation 深度信息转伪3D

> [64]  Y. Wang, W. Chao, D. Garg, B. Hariharan, M. Campbell, and K. Q.Weinberger, “Pseudo-lidar from visual depth estimation: Bridging the gap in 3d object detection for autonomous driving,” 2019.
>
> [9] J. Ku, M. Mozifian, J. Lee, A. Harakeh, and S. L. Waslander, “Joint 3d proposal generation and object detection from view aggregation,” in *International Conference on Intelligent Robots and Systems (IROS)*, 2018, pp. 1–8.
>
> [65]  C. R. Qi, W. Liu, C. Wu, H. Su, and L. J. Guibas, “Frustum pointnets for 3d object detection from RGB-D data,” in *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018, pp. 918–927.

昂贵的、基于LiDAR的3D目标检测技术高度准确，而便宜的、基于图像的3D目标检测执行的准确性则大大降低。 Wang等人[64]认为，弥补这一差距只能通过3D信息的表示，找到了这两种技术之间的差距。

他们提议将基于图像的深度图转换为伪LiDAR表示形式，因为它模仿了LiDAR信号。

最后，他们利用现有的基于LiDAR的3D目标检测pipeline来检测3D目标。通过将3D深度表示转换为伪LiDAR，他们在基于图像的3D目标检测技术的准确性方面获得了空前的性能提升。

### 6. 2D驱动的3D [66]

> [66]  J. Lahoud and B. Ghanem, “2d-driven 3d object detection in rgb- d images,” in *Proceedings of the IEEE International Conference on* Computer Vision (ICCV)*, 2017, pp. 4632–4640.
>
> [40] Z. Deng and L. J. Latecki, “Amodal detection of 3d objects: Inferring 3d bounding boxes from 2d ones in rgb-depth images,” in *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 398–406.
>
> [67] M. M. Rahman, Y. Tan, J. Xue, L. Shao, and K. Lu, “3D object detection: Learning 3d bounding boxes from scaled down 2d bounding boxes in rgb-d images,” *Information Sciences*, vol. 476, pp. 147–158,
>  2019.

[66] proposes a 2D driven 3D object detection method to reduce the search space for objects in 3D,

根据上下文信息执行2D对象检测，3D对象方向回归和对象精炼 object reﬁnement。



[40] also propose a 3D object detection technique by inferring 3D bounding boxes from 2D.从2D推断3D

[40]并没有集成架构，而是两阶段分部计算

1. 得到2D候选和分割蒙板；
2. 计算3D的分类和预测框的回归；



计算2D提议和分割掩码需要相当高的计算成本，并且对于自动化系统是不可行的。另外，他们需要在训练和测试阶段都提供对象分割信息，以预测3D边界框。

[67] Rahman等提出一种多模态区域候选网络，集成方式扩展了[40]，以生成区域候选和一种“膨胀的2D边界框dilated 2D bounding box”方法来生成3D边界框。

## Conclusion

便宜、经济、灵活、2D图像以像素强度的形式提供对象的丰富颜色和纹理信息。
但是，2D图像的缺点是缺少深度信息，这对于准确的对象大小和位置估计（尤其是在弱光条件下）以及检测远处和被遮挡的对象而言是必不可少的。