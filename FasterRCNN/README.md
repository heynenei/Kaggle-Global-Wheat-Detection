# Mobile_v2 FasterRCNN


>目标检测对象微调：
> 
>https://pytorch.apachecn.org/docs/1.2/intermediate/torchvision_tutorial.html

backbone使用ResNet18感觉感受野太大了，使用Mobilev2，感受野为：

$$1+2*4)*2+1+6*2)*2+1+2*2)*2+1+2*2)*2+1+2)*2+1+2)*2+1$$

## model

kaggle运行model计算时一直报错，下载各种torch版本都不行

把roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0])改掉后就可以运行了。（真玄学）

```
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
backbone.out_channels = 1280
# a = torch.Tensor(np.zeros((1,3,1024,1024)))
# b = backbone(a)
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0','1'],
                                                output_size=7,
                                                sampling_ratio=2)
model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)
                   
```

## 运行记录

| 版本 | 单一变量                    | 运行结果 |
| ---- | --------------------------- | -------- |
| v1   | num_epochs = 2              |          |
| v2   | num_epochs = 5,batch_size=4 |          |
|      |                             |          |



# kaggle运行的坑

## CUDA内存溢出

restart session，kaggle没自动释放内存，手动释放后就可以啦

## notebook环境配置

查了好多博客都没有解决方案，在别人分享的notebook中看到了这段代码：

```
!pip install --no-deps '../input/timm-package/timm-0.1.26-py3-none-any.whl' > /dev/null
```

 有些 packages 会依赖一些其它的 package，当我们离线安装 whl 的时候，就无法联网下载依赖包，所以我们需要 ==**--no-deps**== 来去掉依赖包的安装，这样就能离线安装 whl 了 。

 command ==**>/dev/null**==的作用是将是command命令的标准输出丢弃，而标准错误输出还是在屏幕上。 一般来讲标准输出和标准错误输出都是屏幕，因此错误信息还是会在屏幕上输出。这时可以用command >/dev/null 2>&1 这样标准输出与标准错误输出都会被丢弃。1表示标准输出，2表示标准错误输出，2>&1表示将标准错误输出重定向到标准输出。
        >表示输出重定向，如果 command > /usr/log 那其会覆盖log中原来的记录。可以使用>>输出重定向来向文件尾部增加输出记录。














# 附录：

## mobilev2 children()如下：

```

[i for i in backbone.children()]
Out[7]: 
[Sequential(
   (0): ConvBNReLU(
     (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
     (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     (2): ReLU6(inplace=True)
   )
   (1): InvertedResidual(
     (conv): Sequential(
       (0): ConvBNReLU(
         (0): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=32, bias=False)
         (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (1): Conv2d(32, 16, kernel_size=(1, 1), stride=(1, 1), bias=False)
       (2): BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     )
   )
   (2): InvertedResidual(
     (conv): Sequential(
       (0): ConvBNReLU(
         (0): Conv2d(16, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
         (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (1): ConvBNReLU(
         (0): Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=96, bias=False)
         (1): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (2): Conv2d(96, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
       (3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     )
   )
   (3): InvertedResidual(
     (conv): Sequential(
       (0): ConvBNReLU(
         (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
         (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (1): ConvBNReLU(
         (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=144, bias=False)
         (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (2): Conv2d(144, 24, kernel_size=(1, 1), stride=(1, 1), bias=False)
       (3): BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     )
   )
   (4): InvertedResidual(
     (conv): Sequential(
       (0): ConvBNReLU(
         (0): Conv2d(24, 144, kernel_size=(1, 1), stride=(1, 1), bias=False)
         (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (1): ConvBNReLU(
         (0): Conv2d(144, 144, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=144, bias=False)
         (1): BatchNorm2d(144, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (2): Conv2d(144, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
       (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     )
   )
   (5): InvertedResidual(
     (conv): Sequential(
       (0): ConvBNReLU(
         (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
         (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (1): ConvBNReLU(
         (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
         (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (2): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
       (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     )
   )
   (6): InvertedResidual(
     (conv): Sequential(
       (0): ConvBNReLU(
         (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
         (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (1): ConvBNReLU(
         (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=192, bias=False)
         (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (2): Conv2d(192, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
       (3): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     )
   )
   (7): InvertedResidual(
     (conv): Sequential(
       (0): ConvBNReLU(
         (0): Conv2d(32, 192, kernel_size=(1, 1), stride=(1, 1), bias=False)
         (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (1): ConvBNReLU(
         (0): Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=192, bias=False)
         (1): BatchNorm2d(192, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (2): Conv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
       (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     )
   )
   (8): InvertedResidual(
     (conv): Sequential(
       (0): ConvBNReLU(
         (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
         (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (1): ConvBNReLU(
         (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
         (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (2): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
       (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     )
   )
   (9): InvertedResidual(
     (conv): Sequential(
       (0): ConvBNReLU(
         (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
         (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (1): ConvBNReLU(
         (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
         (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (2): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
       (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     )
   )
   (10): InvertedResidual(
     (conv): Sequential(
       (0): ConvBNReLU(
         (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
         (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (1): ConvBNReLU(
         (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
         (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (2): Conv2d(384, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
       (3): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     )
   )
   (11): InvertedResidual(
     (conv): Sequential(
       (0): ConvBNReLU(
         (0): Conv2d(64, 384, kernel_size=(1, 1), stride=(1, 1), bias=False)
         (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (1): ConvBNReLU(
         (0): Conv2d(384, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=384, bias=False)
         (1): BatchNorm2d(384, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (2): Conv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
       (3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     )
   )
   (12): InvertedResidual(
     (conv): Sequential(
       (0): ConvBNReLU(
         (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
         (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (1): ConvBNReLU(
         (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
         (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (2): Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
       (3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     )
   )
   (13): InvertedResidual(
     (conv): Sequential(
       (0): ConvBNReLU(
         (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
         (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (1): ConvBNReLU(
         (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=576, bias=False)
         (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (2): Conv2d(576, 96, kernel_size=(1, 1), stride=(1, 1), bias=False)
       (3): BatchNorm2d(96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     )
   )
   (14): InvertedResidual(
     (conv): Sequential(
       (0): ConvBNReLU(
         (0): Conv2d(96, 576, kernel_size=(1, 1), stride=(1, 1), bias=False)
         (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (1): ConvBNReLU(
         (0): Conv2d(576, 576, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), groups=576, bias=False)
         (1): BatchNorm2d(576, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (2): Conv2d(576, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
       (3): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     )
   )
   (15): InvertedResidual(
     (conv): Sequential(
       (0): ConvBNReLU(
         (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
         (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (1): ConvBNReLU(
         (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
         (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (2): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
       (3): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     )
   )
   (16): InvertedResidual(
     (conv): Sequential(
       (0): ConvBNReLU(
         (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
         (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (1): ConvBNReLU(
         (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
         (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (2): Conv2d(960, 160, kernel_size=(1, 1), stride=(1, 1), bias=False)
       (3): BatchNorm2d(160, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     )
   )
   (17): InvertedResidual(
     (conv): Sequential(
       (0): ConvBNReLU(
         (0): Conv2d(160, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
         (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (1): ConvBNReLU(
         (0): Conv2d(960, 960, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=960, bias=False)
         (1): BatchNorm2d(960, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
         (2): ReLU6(inplace=True)
       )
       (2): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
       (3): BatchNorm2d(320, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     )
   )
   (18): ConvBNReLU(
     (0): Conv2d(320, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False)
     (1): BatchNorm2d(1280, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
     (2): ReLU6(inplace=True)
   )
 ),
 Sequential(
   (0): Dropout(p=0.2, inplace=False)
   (1): Linear(in_features=1280, out_features=1000, bias=True)
 )]

```

## v2 log

```
Iteration #50 loss: 1.2924003601074219
Iteration #100 loss: 1.3207688331604004
Iteration #150 loss: 1.2539582252502441
Iteration #200 loss: 1.0710512399673462
Iteration #250 loss: 0.9630039930343628
Iteration #300 loss: 0.8754738569259644
Iteration #350 loss: 0.8319554924964905
Iteration #400 loss: 1.1836864948272705
Iteration #450 loss: 1.1589711904525757
Iteration #500 loss: 1.0724889039993286
Iteration #550 loss: 1.1808078289031982
Iteration #600 loss: 0.9722001552581787
Iteration #650 loss: 1.1644439697265625
Epoch #0 loss: 1.0876469359440148
Iteration #700 loss: 0.8111879825592041
Iteration #750 loss: 0.9662195444107056
Iteration #800 loss: 0.7937059998512268
Iteration #850 loss: 1.0222702026367188
Iteration #900 loss: 0.849983811378479
Iteration #950 loss: 0.7653313875198364
Iteration #1000 loss: 0.7563380599021912
Iteration #1050 loss: 1.1761846542358398
Iteration #1100 loss: 0.9182501435279846
Iteration #1150 loss: 0.9255369901657104
Iteration #1200 loss: 0.8402021527290344
Iteration #1250 loss: 0.5085845589637756
Iteration #1300 loss: 0.9213157892227173
Iteration #1350 loss: 0.6949498057365417
Epoch #1 loss: 0.8852666549087451
Iteration #1400 loss: 0.6382803916931152
Iteration #1450 loss: 1.0461143255233765
Iteration #1500 loss: 0.9090939164161682
Iteration #1550 loss: 0.8359819650650024
Iteration #1600 loss: 0.8169183731079102
Iteration #1650 loss: 0.7387934327125549
Iteration #1700 loss: 0.48437219858169556
Iteration #1750 loss: 0.9703689813613892
Iteration #1800 loss: 0.7435070872306824
Iteration #1850 loss: 0.774308443069458
Iteration #1900 loss: 1.4260516166687012
Iteration #1950 loss: 0.7171157002449036
Iteration #2000 loss: 0.4673672020435333
Epoch #2 loss: 0.8082631008487684
Iteration #2050 loss: 0.9568806886672974
Iteration #2100 loss: 0.9023582339286804
Iteration #2150 loss: 0.7650253176689148
Iteration #2200 loss: 0.9549641013145447
Iteration #2250 loss: 0.8457344770431519
Iteration #2300 loss: 0.800534188747406
Iteration #2350 loss: 0.667407214641571
Iteration #2400 loss: 1.1296151876449585
Iteration #2450 loss: 0.8486276865005493
Iteration #2500 loss: 0.8028806447982788
Iteration #2550 loss: 0.7199442386627197
Iteration #2600 loss: 0.7345008254051208
Iteration #2650 loss: 0.7195303440093994
Iteration #2700 loss: 0.6429624557495117
Epoch #3 loss: 0.7620894248587694
Iteration #2750 loss: 0.5161945223808289
Iteration #2800 loss: 0.8676239252090454
Iteration #2850 loss: 0.7502480745315552
Iteration #2900 loss: 0.9182735681533813
Iteration #2950 loss: 0.6842472553253174
Iteration #3000 loss: 0.7219998240470886
Iteration #3050 loss: 0.3948180079460144
Iteration #3100 loss: 0.8459411859512329
Iteration #3150 loss: 0.7018390893936157
Iteration #3200 loss: 0.7666540145874023
Iteration #3250 loss: 0.7135879397392273
Iteration #3300 loss: 0.5241468548774719
Iteration #3350 loss: 0.415126770734787
Epoch #4 loss: 0.7269600099351593
```

