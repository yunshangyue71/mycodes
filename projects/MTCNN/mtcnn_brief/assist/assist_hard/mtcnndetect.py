#!/usr/bin/env python3
# -*- coding: utf-8 -*- 
import numpy as np
import cv2

from assist.assist_hard.convert_to_square import convert_to_square

class MtcnnDetector(object):
    def __init__(self, detectors, min_face_size, stride,
                 threshold, scale_factor):
        self.pnet_detector = detectors[0]
        self.rnet_detector = detectors[1]
        self.onet_detector = detectors[2]
        self.min_face_size = min_face_size
        self.stride = stride
        self.thresh = threshold
        self.scale_factor = scale_factor
    def detect_face(self, iter_im):
        print('###########数据读入完毕开始检测###################')
        allimages_boxes = []
        allimages_marks = []
        image_done_count = 0#图片计数
        for im in iter_im:
            if self.pnet_detector:
                boxes, boxes_c, mark = self.detect_pnet(im)
                if boxes_c is None:
                    allimages_boxes.append(np.array([]))
                    allimages_marks.append(np.array([]))
                    continue
            if self.rnet_detector:
                boxes,boxes_c,mark = self.detect_rnet(im, boxes_c)
                if boxes_c is None:
                    allimages_boxes.append(np.array([]))
                    allimages_marks.append(np.array([]))
                    continue
            if self.onet_detector:
                boxes, boxes_c,mark =self.detect_onet(im, boxes_c)
                if boxes_c is None:
                    allimages_boxes.append(np.array([]))
                    allimages_marks.append(np.array([]))
                    continue
            allimages_boxes.append(boxes_c)
            #mark = [1]
            allimages_marks.append(mark)
            image_done_count += 1  
            if image_done_count%10 == 0:
                print('\r>>%d images detect done'%(image_done_count),end = '')
        return allimages_boxes, allimages_marks
    
    def detect_pnet(self, im):
        h,w,c = im.shape
        net_size = 12
        current_scale = float(net_size)/self.min_face_size
        im_resized = self.scaled_image(im,current_scale)
        current_height, current_width, _ =im_resized.shape
        im_allscales_boxes = list()
        while min(current_height, current_width)> net_size:
            cls_prob, box_pred  = self.pnet_detector.predict(im_resized)#执行P_Net
            #print(cls_prob,'p')
            clsgenbox_clsscore_boxpred = self.slt_cls_clsgen_box(
                    cls_prob[:,:,1],box_pred,current_scale,self.thresh[0])#根据cls预测box
            current_scale *= self.scale_factor
            im_resized = self.scaled_image(im,current_scale)
            current_height, current_width, _=im_resized.shape
            if clsgenbox_clsscore_boxpred.size == 0:#一次scale图片前向传播后，score没有大于阈值的
                continue
            keep = py_nms(clsgenbox_clsscore_boxpred[:,:5], 0.5,'Union')#剔除一个scale相似的
            clsgenbox_clsscore_boxpred = clsgenbox_clsscore_boxpred[keep]
            im_allscales_boxes.append(clsgenbox_clsscore_boxpred)
        if len(im_allscales_boxes) == 0:
            return None,None,None
        im_allscales_boxes = np.vstack(im_allscales_boxes)
        keep = py_nms(im_allscales_boxes[:,0:5],0.7,'Union')
        im_allscales_boxes = im_allscales_boxes[keep]
        boxes = im_allscales_boxes[:,:5]
        bbw = im_allscales_boxes[:,2] - im_allscales_boxes[:, 0] + 1
        bbh = im_allscales_boxes[:,3] - im_allscales_boxes[:, 1] + 1
       
        boxes_c = np.vstack([im_allscales_boxes[:,0] + im_allscales_boxes[:,5] *bbw,
                             im_allscales_boxes[:,1] + im_allscales_boxes[:,6] *bbh,
                             im_allscales_boxes[:,2] + im_allscales_boxes[:,7] *bbw,
                             im_allscales_boxes[:,3] + im_allscales_boxes[:,8] *bbh,
                             im_allscales_boxes[:,4] 
                             ])
        boxes_c = boxes_c.T
        return boxes,boxes_c,None
   
    def detect_rnet(self, im, dets):
        im_h,im_w,im_c = im.shape
        dets = convert_to_square(dets)
        dets[:,0:4] = np.round(dets[:,0:4])
        [dy1,dy2, dx1, dx2, 
        cls_prob_square_y1, cls_prob_square_y2,
        cls_prob_square_x1, cls_prob_square_x2,
        cls_prob_square_w, cls_prob_square_h]=self.box_fit_in_image(dets,im_w,im_h)#获取dets的信息还正方形后的与img的比较关系
        num_boxes = dets.shape[0]#一张图片里面有多少个bbox
        cropped_ims = np.zeros((num_boxes, 24,24,3),dtype = np.float32)
        for i in range(num_boxes):#将每个框在原图中的照片截取出来，并整形成24,24
            tmp = np.zeros((cls_prob_square_h[i], cls_prob_square_w[i], 3),dtype = np.uint8)
            tmp[dy1[i]:dy2[i] + 1, dx1[i]:dx2[i]+1,:] = im[cls_prob_square_y1[i]:cls_prob_square_y2[i]+1,
                                                            cls_prob_square_x1[i]:cls_prob_square_x2[i]+1,:]
            cropped_ims[i,:,:,:] =(cv2.resize(tmp,(24,24)) -127.5)/128
        cls_scores, box_pred,_ = self.rnet_detector.predict(cropped_ims)#一个图片的所有框截取的图片进行detector
        cls_scores = cls_scores[:,1]
        keep_inds = np.where(cls_scores>self.thresh[1])[0]
        if len(keep_inds)>0:
            boxes = dets[keep_inds]
            boxes[:,4] = cls_scores[keep_inds]#更新得分值
            box_pred = box_pred[keep_inds]
        else:
            return None,None,None
        keep = py_nms(boxes,0.6)
        boxes = boxes[keep]
        boxes_c = self.calibrate_box(boxes,box_pred[keep])
        return boxes, boxes_c, None
            
    def detect_onet(self,im,dets):
        im_h,im_w,im_c = im.shape
        dets = convert_to_square(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])
        [dy1,dy2, dx1, dx2, 
        cls_prob_square_y1, cls_prob_square_y2,
        cls_prob_square_x1, cls_prob_square_x2,
        cls_prob_square_w, cls_prob_square_h]=self.box_fit_in_image(dets,im_w,im_h)#获取dets的信息还正方形后的与img的比较关系
        num_boxes = dets.shape[0]#一张图片里面有多少个bbox
        cropped_ims = np.zeros((num_boxes, 48, 48, 3), dtype=np.float32)
        for i in range(num_boxes):
            tmp = np.zeros((cls_prob_square_h[i], cls_prob_square_w[i], 3),dtype = np.uint8)
            tmp[dy1[i]:dy2[i] + 1, dx1[i]:dx2[i]+1,:] = im[cls_prob_square_y1[i]:cls_prob_square_y2[i]+1,
                                                            cls_prob_square_x1[i]:cls_prob_square_x2[i]+1,:]
            cropped_ims[i, :, :, :] = (cv2.resize(tmp, (48, 48)) - 127.5) / 128
        cls_scores, box_pred, mark_pred = self.onet_detector.predict(cropped_ims)
        cls_scores = cls_scores[:, 1]
        keep_inds = np.where(cls_scores > self.thresh[2])[0]
        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            boxes[:, 4] = cls_scores[keep_inds]
            box_pred = box_pred[keep_inds]
            #print('box_pred\n',box_pred)
            mark_pred = mark_pred[keep_inds]
            #print('mark_pred\n',mark_pred)
        else:
            return None, None, None
        
        boxes_c = self.calibrate_box(boxes, box_pred)
        
        w = boxes[:, 2] - boxes[:, 0] + 1
        h = boxes[:, 3] - boxes[:, 1] + 1
        
        mark_pred[:, 0::2] = (np.tile(w, (5, 1)) * mark_pred[:, 0::2].T + np.tile(boxes[:, 0], (5, 1)) - 1).T
        mark_pred[:, 1::2] = (np.tile(h, (5, 1)) * mark_pred[:, 1::2].T + np.tile(boxes[:, 1], (5, 1)) - 1).T
        #boxes_c = self.calibrate_box(boxes, box_pred)

        boxes = boxes[py_nms(boxes, 0.6, "Minimum")]
        keep = py_nms(boxes_c, 0.6, "Minimum")
        boxes_c = boxes_c[keep]
        mark_pred = mark_pred[keep]
        return boxes, boxes_c, mark_pred

    """
    将img按着scale的比例缩小，并将0-255的值改为（(1,1）的范围
    """
    def scaled_image(self, img, scale):
        height, width, channels = img.shape
        new_height = int(height *scale)
        new_width = int(width * scale)
        new_dim = (new_width, new_height)
        img_scaled = cv2.resize(img, new_dim,interpolation = cv2.INTER_LINEAR)
        img_scaled = (img_scaled -127.5)/128
        return img_scaled
    def slt_cls_clsgen_box(self, cls_prob, box_pred, current_scale,threshold):
        """select score ,cls gennerate box只在detect_pnet中使用，用于产生box"""
        stride = 2
        cellsize = 12
        cls_select = np.where(cls_prob > threshold)
        if cls_select[0].size == 0:
            return np.array([])
        dx1,dy1,dx2,dy2 = [box_pred[cls_select[0],cls_select[1], i] for i in range(4)]
        box_pred = np.array([dx1, dy1,dx2,dy2])
        cls_score = cls_prob[cls_select[0],cls_select[1]]#这个是人脸的判断的分数
        boundingbox = np.vstack([np.round((stride*cls_select[1])/current_scale),
                                 np.round((stride*cls_select[0])/current_scale),
                                 np.round((stride*cls_select[1] + cellsize)/current_scale),
                                 np.round((stride*cls_select[0] + cellsize)/current_scale),
                                 cls_score,
                                 box_pred ])
    #前向传播二道的reg，是随机产生的框相对于label框的
        return boundingbox.T
    def box_fit_in_image(self,cls_prob_squared_boxes,im_w,im_h):
        cls_prob_square_w = cls_prob_squared_boxes[:,2] - cls_prob_squared_boxes[:,0]+1#cls_pred_box的宽高
        cls_prob_square_h = cls_prob_squared_boxes[:,3] - cls_prob_squared_boxes[:,1]+1
        
        cls_prob_square_x1,cls_prob_square_y1 = cls_prob_squared_boxes[:,0],cls_prob_squared_boxes[:,1]#p1坐标
        cls_prob_square_x2,cls_prob_square_y2 = cls_prob_squared_boxes[:,2],cls_prob_squared_boxes[:,3]#p2坐标
        
        num_box = cls_prob_squared_boxes.shape[0]
        dx1 = np.zeros((num_box,))                                  #x1小于0，cls_prob_box_x1为0
        tmp_index = np.where(cls_prob_square_x1<0)                  #dx1记录x1需要增加多少
        dx1[tmp_index] = 0 - cls_prob_square_x1[tmp_index]
        cls_prob_square_x1[tmp_index] = 0
        
        dy1 = np.zeros((num_box,))
        tmp_index = np.where(cls_prob_square_y1<0)                  #y1小于0，cls_prob_box_y1为0
        dy1[tmp_index] = 0 -cls_prob_square_y1[tmp_index]            #dy1记录y1需要增加多少
        cls_prob_square_y1[tmp_index] = 0
        
        dx2 = cls_prob_square_w.copy() - 1        
        tmp_index = np.where(cls_prob_square_x2 > im_w-1)
        dx2[tmp_index] = cls_prob_square_w[tmp_index] + im_w -2 -cls_prob_square_x2[tmp_index]
        cls_prob_square_x2[tmp_index] = im_w - 1
        
        dy2 = cls_prob_square_h.copy() -1
        tmp_index = np.where(cls_prob_square_y2>im_h-1)
        dy2[tmp_index] = cls_prob_square_h[tmp_index] + im_h -2 -cls_prob_square_y2[tmp_index]
        cls_prob_square_y2[tmp_index] = im_h -1
      
        return_list = [dy1,dy2, dx1, dx2, 
                       cls_prob_square_y1, cls_prob_square_y2,
                       cls_prob_square_x1, cls_prob_square_x2,
                       cls_prob_square_w, cls_prob_square_h]
        return_list = [item.astype(np.int32) for item in return_list]
        return return_list
    def calibrate_box(self, dets, box_pred_keep):
        bbox_c = dets.copy()
        w = dets[:,2] - dets[:,0] + 1
        w = np.expand_dims(w,1)
        h = dets[:,3] - dets[:,1] + 1
        h = np.expand_dims(h, 1)
        reg_m = np.hstack([w, h,w, h])
        aug = reg_m * box_pred_keep
        bbox_c[:, 0:4] = bbox_c[:,0:4] + aug
        return bbox_c
    
'''剔除比较相似的图框'''           
def py_nms(dets, thresh,mode = 'Union'):
    
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]#排序
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]#由大到小排序
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1: ]])
        yy1 = np.maximum(y1[i], y1[order[1: ]])
        xx2 = np.minimum(x2[i], x2[order[1: ]])
        yy2 = np.minimum(y2[i], y2[order[1: ]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w* h
        if mode == 'Union':
            #重叠率=a + b - 重叠部分的 
            ovr = inter/(areas[i] + areas[order[1:]] - inter)
        elif mode == 'Minimum':
            ovr = inter/np.minimum(areas[i],areas[order[1:]])
        inds = np.where(ovr <= thresh)[0]#[0]
        #inds是order[1：]中l留下来的的索引
        #在order中对应留下来的索引序号要多1
        order = order[inds + 1]
    return keep