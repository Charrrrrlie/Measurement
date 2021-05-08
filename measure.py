import cv2
import numpy as np
import math

ix,iy,bx,by=0,0,0,0
res_loc=[]
bbox=[]

#写的太不优雅 (x
def centering(img,ix,iy,bx,by):

    gray_img = cv2.cvtColor(img[iy:by,ix:bx], cv2.COLOR_RGB2GRAY)

    #平滑
    cv2.medianBlur(gray_img,3,gray_img)

    # 大津法
    _, binary = cv2.threshold(gray_img, 0, 255, cv2.THRESH_OTSU, 1)
    # 反色
    inv_binary = binary
    cv2.bitwise_not(binary, inv_binary)

    pre_process=inv_binary.copy()

    # contours,_=cv2.findContours(inv_binary,cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    #
    # #全部连通区域
    # for i in range(len(contours)):
    #
    #     #面积
    #     area=cv2.contourArea(contours[i])
    #     #圆度
    #     a = area * 4 * math.pi
    #     b = cv2.arcLength(contours[0], True)**2
    #     print(a/b)
    #     if area>400:
    #         cv2.drawContours(img, contours[i], -1, (0, 0, 0), thickness=-1)
    #
    # center_x=contours[max_idx].squeeze()[:,0].mean()
    # center_y=contours[max_idx].squeeze()[:,1].mean()

    num_objects, labels = cv2.connectedComponents(inv_binary,connectivity=8)

    area={}
    for y in range(len(labels)):
        for x in range(len(labels[y])):
            if labels[y,x]==0:
                continue
            if area.get(labels[y,x],0):
                area[labels[y,x]].append([y,x])
            else:
                area[labels[y,x]]=[[y,x]]
    #面积约束
    count=0
    max_squre=0
    max_squre_idx=-1
    for i in range(1,num_objects):
        square=len(area[i])
        if square>max_squre:
            max_squre=square
            max_squre_idx=i

        if square<800:  #thred
            for j in area[i]:
                inv_binary[j[0],j[1]]=0
        else:
            count+=1

    #上面循环的硬阈值出问题
    if count==0:
        for j in area[max_squre_idx]:
            inv_binary[j[0], j[1]] = 255
        print('high_threshold')

    #圆度约束
    # if count>1:
    #     contours, _ = cv2.findContours(inv_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #
    #     max_round=0
    #     max_idx=-1
    #     for i in range(len(contours)):
    #         s=cv2.contourArea(contours[i])
    #         #圆度
    #         a = s * 4 * math.pi
    #         b = cv2.arcLength(contours[i], True)**2
    #         print(a,b,a/b)
    #         if b!=0:
    #             if max_round<a/b:
    #                 max_round=a/b
    #                 max_idx=i
    #     print(max_idx)
    #     if max_idx!=-1:
    #         inv_binary=cv2.drawContours(inv_binary,contours,max_idx,(0,255,0),-1)

    #圆度约束2
    if count>1:
        max_val=0
        max_idx=-1
        for i in range(1,num_objects):
            if len(area[i])>800:
                group=np.array(area[i])
                dx=np.max(group[:,1])-np.min(group[:,1])
                dy=np.max(group[:,0])-np.min(group[:,0])

                val=min(dx/dy,dy/dx)
                if val>max_val:
                    max_idx=i

        final=np.array(area[max_idx])
        center_x=np.mean(final[:,1])
        center_y=np.mean(final[:,0])
        list=area[max_idx]

    else:
        size=0
        center_x, center_y=0,0
        list=[]
        for i in range(len(inv_binary)):
            for j in range(len(inv_binary[i])):
                if inv_binary[i][j]!=0:
                    center_y+=i
                    center_x+=j
                    size+=1
                    list.append([i,j])
        center_x/=size
        center_y/=size

    return pre_process,list,center_x,center_y

def draw(event,x,y,flags,params):
    global ix,iy,bx,by,res_loc
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE and flags==cv2.EVENT_FLAG_LBUTTON:

        img2 = plt_img.copy()  #实时更新
        cv2.rectangle(img2, (ix, iy), (x, y), (0, 0, 255), 3)
        cv2.imshow('Measurement', img2)
    elif event == cv2.EVENT_LBUTTONUP:
        bx, by = x, y
        #中心点
        temp_img,target_list,cx,cy=centering(img,ix,iy,bx,by)
        int_cx=int(cx)
        int_cy=int(cy)

        temp_img=temp_img.reshape(temp_img.shape[0],temp_img.shape[1],1)
        temp_img=np.tile(temp_img,(1,1,3))
        cv2.namedWindow('Refine',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Refine',temp_img.shape[1]*3,temp_img.shape[0]*3)
        cv2.imshow('Refine', temp_img)

        #目标区域改为绿色
        for i in target_list:
            temp_img[i[0],i[1],0]=0
            temp_img[i[0],i[1],1]=100
            temp_img[i[0],i[1],2]=0

        while(1):
            show_img=temp_img.copy()
            cv2.line(show_img,(int_cx-10,int_cy),(int_cx+10,int_cy),(255,255,0),1)
            cv2.line(show_img,(int_cx,int_cy-10),(int_cx,int_cy+10), (255, 255, 0), 1)

            # 验证
            cv2.imshow('Refine',show_img)
            k=cv2.waitKey(1)

            #确认
            if k==ord('y'):
                cv2.rectangle(plt_img,(ix,iy),(bx,by),(0,255,0),3)
                cv2.imshow('Measurement',plt_img)
                cv2.destroyWindow('Refine')
                res_loc.append([cx+ix,cy+iy])
                bbox.append([ix,iy,bx,by])
                break
            #退出
            elif k == ord('q'):
                cv2.destroyWindow('Refine')
                break

            #微调
            elif k == ord('w'):
                cy=max(0,cy-1)
                int_cy=int(cy)
            elif k == ord('s'):
                cy=min(temp_img.shape[1],cy+1)
                int_cy=int(cy)
            elif k == ord('a'):
                cx=max(0,cx-1)
                int_cx=int(cx)
            elif k == ord('d'):
                cx=min(temp_img.shape[0],cx+1)
                int_cx=int(cx)

if __name__=='__main__':

    root_path='./photos/1.jpg'
    load_path='./results/test_pts.txt'
    temp_path='./results/pts.txt'
    final_path='./results/measurement.txt'

    img=cv2.imread(root_path)
    print('Please input the measuring type')
    type=input('1:loading 2:RE-measuring:')

    if type=='2':

        plt_img=img.copy()

        cv2.namedWindow('Measurement',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Measurement',img.shape[1]//4,img.shape[0]//4)
        cv2.imshow("Measurement", plt_img)
        cv2.setMouseCallback('Measurement',draw,plt_img)   #回调鼠标
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        #赶快保存量测好的中间结果
        f=open(temp_path,'w')
        for i,j in zip(res_loc,bbox):
            f.write(str(i[0])+" "+str(i[1])+" "+
                    str(j[0])+" "+str(j[1])+" "+str(j[2])+" "+str(j[3])
                    +"\n")

        f.close()

    elif type=='1':
        f=open(load_path,'r')
        data=f.readlines()
        f.close()

        res_loc=[]
        for i in data:
            vals=str(i.split("\n")[:-1][0]).split(" ")
            res_loc.append(list(map(float,vals[0:2])))
            bbox.append(list(map(int,vals[2:6])))
        print(res_loc)

    #输入编号
    res_num=[]
    #num_list=np.arange(1,len(res_loc)+1)
    for i in range(len(res_loc)):
        cv2.namedWindow('Number', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Number', img.shape[1] // 4, img.shape[0] // 4)

        temp=img.copy()
        cv2.rectangle(temp,(bbox[i][0],bbox[i][1]),(bbox[i][2],bbox[i][3]),(0,0,255),3)
        cv2.imshow("Number", temp)
        cv2.waitKey(0)

        num=input('Please input the point number:')
        # num=str(num_list[i])
        if num!='-1':
            xx=int(res_loc[i][0])
            yy=int(res_loc[i][1])
            cv2.rectangle(img, (bbox[i][0], bbox[i][1]), (bbox[i][2], bbox[i][3]), (0,255,0),3)
            cv2.line(img, (xx- 25, yy), (xx + 25, yy), (0, 255, 0),8)
            cv2.line(img, (xx, yy - 25), (xx, yy + 25),(0, 255, 0),8)
            cv2.putText(img,num,(bbox[i][2]+10,bbox[i][3]),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),5)
            res_num.append(num)
        #删除废点
        else:
            del(res_loc[i:i+1])


    cv2.imwrite("./results/result.jpg",img)
    f=open(final_path,'w')

    for i in range(len(res_num)):
        f.write(str(res_num[i])+" "+str(res_loc[i][0])+" "+str(res_loc[i][1])+"\n")

    f.close()


