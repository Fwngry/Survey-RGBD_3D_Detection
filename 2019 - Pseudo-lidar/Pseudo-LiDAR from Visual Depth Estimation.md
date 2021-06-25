> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/wqwqqwqw1231/article/details/103300948)

### 文章目录

*   [检测结构](#_7)
*   [对使用 Pseudo LiDAR 有效性的解释](#Pseudo_LiDAR_17)
*   [实验](#_41)

2019CVPR

本文提出了一个重要观点：**使用图像做三维目标检测，其效果差不是因为使用图像得到的深度信息不准确，而是因为使用前视图这种表示方式的问题**。

由于本文并未提出什么新的网络，所以这篇 paper 的解读与其他的结构不太一样。

检测结构
====

![](https://img-blog.csdnimg.cn/20191128205420649.png)  
上图为本文提出的检测结构，整体分为两步走，第一步通过计算 Depth Map，恢复出 Pseudo LiDAR，第二部使用融合图像和点云的方法检测三维物体。

Depth Map -> Pseudo LiDAR：  
![](https://img-blog.csdnimg.cn/20191128205652702.png#pic_center)  
说白了就是将深度图 (u, v, d) 的表示方式转为点云的表示方式(x, y, z)。

然后就是对 Pseudo LiDAR 的处理，就是丢掉一些超出一定 x，y，z 边界的点，例如高于激光雷达 1m 的点丢弃掉等。

对使用 Pseudo LiDAR 有效性的解释
=======================

这块内容主要是在 Data representation matters 这一节中讲的，其实我对这一节的解释很不认可。

首先先说一下文中的解释：  
文中首先提到了 convolution 有两个假设：

*   local neighborhoods in the image have meaning, and the network should look at local patches
*   all neighborhoods can be operated upon in an identical manner

然后提到在前视图中其实并不满足这两个假设：

*   local patches on 2D images are only coherent physically if they are entirely contained in a single object. If they straddle object boundaries, then two pixels can be co-located next to each other in the depth map, yet can be very far away in 3D space. 二维图像上的局部面片只有在完全包含在单个对象中时才是物理相干的。如果它们跨越对象边界，那么两个像素可以在深度图中彼此相邻，但在三维空间中却可能非常遥远。
*   objects that occur at multiple depths project to different scales in the depth map. A similarly sized patch might capture just a side-view mirror of a nearby car or the entire body of a far-away car. Existing 2D object detec- tion approaches struggle with this breakdown of assumptions and have to design novel techniques such as feature pyramidsto deal with this challenge. 出现在多个深度的对象在深度图中投影到不同的比例。同样大小的 patch 可能只捕捉到附近汽车的侧视镜或远处汽车的整个车身。现有的 2D 对象检测方法与这种假设的分解相抗衡，必须设计新颖的技术，如特征金字塔来应对这一挑战。

我觉得第一个解释有点牵强：是因为在处理图像时，局部的 patch 如果在边缘上，其像素 intensity 的距离可能很大。所以 CNN 能够处理纹理边缘，为什么不能处理深度边缘？  
第二点的解释其实就是说，俯视图的好处是要预测的 box 的大小与车辆的远近没有关系。但这种 scale 的变化在图像中就存在，需要用 FPN 的方法去弥补。

**所以我认为，文章中想强调的点就是在 Point Cloud 中检测，或者在俯视图中检测可以带来不用处理 scale 的好处。而这一点正是巨大涨点的原因！**

然后作者在本节也提供了一个实验，但我并不觉得这个实验有什么用。图片中，车辆与背景存在遮挡，车辆与背景的深度相差很大，在边缘做个滤波，简单的理解就是平均一下，很容易就将边缘的点拉的离真实很远，这个事实很好理解。但我还是认为，CNN 能处理纹理边缘，同样能处理深度边缘。

我认为本文合理的解释是在 Experiment 这个部分中：  
_“while performing local computations like convolutions or ROI pooling along height and width of an image makes sense to 2D object detection, it will oper- ate on 2D pixel neighborhoods with pixels that are far apart in 3D, making the precise localization of 3D objects much harder”_  
我的理解是：在对 box 进行预测时，2D 中的 anchor 或者 proposal 中包含了很多 neighborhood 的信息，但这些图像空间内的 neighborhood 在三维空间中离得很远，所以会产生很大的干扰。

实验
==

本文的实验做的非常充分。总体来说就是先找一些 baselines, 使用图像做三维检测方法，MONO3D [4], 3DOP [5], and MLF [33]。

然后使用 PSMNET [3], DISPNET [21], and SPS-STEREO [35] 得到深度图，将其转为 Pseudo LiDAR，再用 F-PointNet 和 AVOD 去检测，得到的结果与 baseline 做比较。效果要好很多。

实验结果拿到了使用图像做三维目标检测的第一，效果非常好。但我觉得这个实验与本文章强调的点还有一些距离：

1、感觉实验没有比摆脱一个问题：是不是有可能 Frustum-PointNet 和 AVOD 的性能就是强！  
因为本文是想强调，使用图像效果差的原因是在数据的表示方面，而不是精度方面。我的分析是说，其实这个差别可能是在在俯视图中检测可以带来不用处理 scale 的好处。所以我认为还可以检测网络相同，不同点只是用俯视图和前视图的实验。例如使用 FasterRCNN 或者 FPN，只是把最后的回归头的回归量的数量由 2D 的 4 个变成 3D 的 7 个（x, y, z, w, h, l, angle-z）

2、另外一点，文章强调：**使用图像做三维目标检测，其效果差不是因为使用图像得到的深度信息不准确，而是因为使用前视图这种表示方式的问题**。但可以看到使用 Pseudo LiDAR 的效果还是要比使用激光雷达差很多的。这个在文章中并没有详细讨论。可以从原文的 Table I 中可以看到，只有 AVOD，IoU=0.5，Easy 情况下，AP 差不多，其他情况下，差的还是挺多的。这是否是由于恢复的三维信息不准确而产生的？我能想到的实验，是用激光点替换一部分用双目视觉恢复的深度点，替换的数量递增，看看是否效果也能递增。从而解释最终结果的 gap。另外一个实验，可以将激光点云投影到前视图，做成深度图，看看效果会下降多少。

总体而言，这篇文章的 idea 很明显，实验效果也很好！