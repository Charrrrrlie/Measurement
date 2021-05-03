## 📏近景摄影测量控制场标志交互式量测程序

### Introduction
**近景摄影测量课间实习内容**

控制场标志点的像点坐标量测，通过交互式框选大致区域，再利用*大津法*自适应二值化、*形态学*约束得到兴趣区域，最终采用重心法确定标志点圆心。

程序示例图如下：
<div align=center><img width="500" height="360" src="https://github.com/Charrrrrlie/Measurement/blob/main/res_demo/demo1.png"/></div>

### Requirements
- Python3
- OpenCV 4.2.0
- Numpy 1.18.5 

### Procedures

```
run measure.py
```
- **选择模式**  1：读取文件 2：重新测量
  
- **重新测量模式**：
  
  鼠标左键单击拖动矩形框，范围由红色矩形框实时绘制；
  
  松开左键，框选范围的二值化和兴趣区域选取结果在新窗口中弹出显示。*深绿色*代表兴趣区域，中心点由*浅蓝色*十字丝标示；
  - `y`键确认选取结果
  - `q`键退出，重新选取
  - `w a s d`逐像素微调中心点位置

  后续像点编号同读取文件模式。

- **读取文件模式**：
  
  根据`load_path`直接进入像点编号步骤；

  按照像点量测顺序/文件存储顺序逐一以红色矩形框显示量测结果，确认编号后关闭窗口即可从键盘输入编号值；

  若输入为`-1`则可删除该点，其他值则确认存储该点。确认存储的像点包围盒、标志中心和编号以绿色显示；

  下一个待编号矩形仍由红色矩形框显示，逐一编号直至全部包围盒被编号完毕即可。


### Path Description
`root_path` 待量测图片路径

`load_path` 读取文件模式下，量测坐标文件读取路径

`temp_path` 重新量测模式下，量测坐标文件存储路径
  
`final_path` 带编号的最终结果文件路径


##
赶ddl写的代码不太优雅，若有其他逻辑错误可以联系本人(charles.yyc@foxmail.com) 

😃 *feel free！*
