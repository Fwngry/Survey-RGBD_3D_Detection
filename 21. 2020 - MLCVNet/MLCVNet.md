In fact, unlike general object detection in open scenes, indoor scenes usually contain strong context constraints, which can be utilized in indoor scene understanding tasks such as 3D object detection.

However, treating every point patch and object individually, VoteNet lacks the consideration of the relationships between different objects and between objects and the scene they belong to, which limits its detection accuracy.

然而，单独处理每一个点修补和物体，VoteNet缺乏对不同物体之间以及物体和它们所属的场景之间关系的考虑，这限制了它的检测精度。

We thus propose a novel 3D object detection framework, called Multi-Level Context VoteNet (MLCVNet), to incorporate into VoteNet multi-level contextual information for 3D object detection. Speciﬁcally, we propose a uniﬁed network to model the multi-level contexts, from local point patches to global scenes. The difference between VoteNet and the proposed network is highlighted in Fig. 2. To model the contextual information, three sub-modules are proposed in the framework, i.e., patch-to-patch context (PPC) module, object-to-object context (OOC) module and the global scene context (GSC) module. In particular, similar to [14], we use the self-attention mechanism to model the relationships between elements in both PPC and OOC modules. These two sub-modules aim at adaptively encoding contextual information at the patch and object levels, respectively. For the scene-level, we design a new branch as shown in Fig. 2(c) to fuse multi-scale features to equip the network with the ability of learning global scene context.

因此，我们提出了一个新的三维物体检测框架，称为多级上下文VoteNet（MLCVNet），将多级上下文信息纳入VoteNet进行三维物体检测。具体来说，我们提出了一个统一的网络来模拟多层次的背景，从局部的点补丁到全球的场景。图2强调了VoteNet和拟议网络之间的区别。为了对上下文信息进行建模，该框架中提出了三个子模块，即补丁到补丁的上下文（PPC）模块，物体到物体的上下文（OOC）模块和全球场景上下文（GSC）模块。特别是，与[14]类似，我们使用自我关注机制来模拟PPC和OOC模块中元素之间的关系。这两个子模块的目的是分别在斑块和物体层面上对背景信息进行自适应编码。对于场景层面，我们设计了一个新的分支，如图2(c)所示，以融合多尺度特征，使网络具备学习全局场景的能力。

