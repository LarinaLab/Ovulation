clear all;
clc;

fileDirectory='\\ent-res2-p01\bcm-mpb-larinalab\Kohei\042623_Vivo_hCG11h_BeforeOvulation-AfterOvulation\Raw Data\042623_Vivo_hCG13h_500x500_3x5_Bscan8p353\images\1\';
fileName1='image_';
fileName2='.tiff';

frameNumber=500;
width=500;
depth=1024;

%% volwise segmentation
% for series_num=250:310
%     dataMatrix=imread(strcat(fileDirectory,fileName1,num2str(series_num,'%06g'),fileName2));
%     Vol(:,:,series_num-249)=dataMatrix(:,:);
% end 
% 
% BW=imbinarize(Vol,'adaptive',sensitivity=0.5);
% invbinvol=1-BW;
% imshow(invbinvol(:,:,30));

%% frame wise Multilevel image segmentation
% I=imread(strcat(fileDirectory,fileName1,num2str(280,'%06g'),fileName2));
% thresh = multithresh(I,2);
% labels = imquantize(I,thresh);
% labelsRGB = label2rgb(labels);
% imshow(labelsRGB)
% title("Segmented Image")

%% frame wise adaptthresh (regional) image segmentation
% I=imread(strcat(fileDirectory,fileName1,num2str(280,'%06g'),fileName2));
% T = adaptthresh(I, 0.2);
% BW = imbinarize(I,T);
% figure
% imshowpair(I, BW, 'montage')

%% active contour front and back ground // not suitable because it requires manually name the region
% I=imread(strcat(fileDirectory,fileName1,num2str(280,'%06g'),fileName2));
% mask = zeros(size(I));
% mask(350:370,180:200) = 1;
% 
% bw = activecontour(I,mask,800);
% imshow(bw)
% title('Segmented Image, 800 Iterations')

%% imsegfm // good for specific frame need to specify starting point and adjust threshold 
% for i=250:310
% I=imread(strcat(fileDirectory,fileName1,num2str(i,'%06g'),fileName2));
% mask = false(size(I)); 
% mask(369,185) = true;
% W = graydiffweight(I, mask, 'GrayDifferenceCutoff', 25);
% thresh = 0.017;
% [BW, D] = imsegfmm(W, mask, thresh);
% imwrite(BW,strcat('F:\code for kohei ovulation project\imsegfm\imsegfm_Z',num2str(i-249),'.tif'));
% end 

%% imsegfm3D // cutoff 40 thresh 0.04 works nicely // different volume requires different threshold ..
% % ovulation_vol(1:200)=0;
% mkdir(strcat('F:\code for kohei ovulation project\imsegfm40_gradient'));
% for j = 1:200
% for i= 251:300
% I=imread(strcat(fileDirectory,fileName1,num2str(i+(j-1)*500,'%06g'),fileName2));
% BW=imresize(I, [1024 1500]);
% Vol(:,:,i-250)=BWc;
% end 
% 
% mask = false(size(Vol)); 
% mask(140,120,29) = true;
% %W = graydiffweight(Vol, mask, 'GrayDifferenceCutoff',40);
% W = gradientweight(Vol,3,'RolloffFactor',3,'WeightCutoff',0.1);
% thresh =0.1;
% BW2 = imsegfmm(W, mask, thresh);
%  for i=29
% imwrite(BW2(:,:,i),strcat('F:\code for kohei ovulation project\imsegfm40_gradient\imsegfm_Z',num2str(i),'_T',num2str(j),'.tif'));
% end 
% %  ovulation_vol(j)=length(find(BW2==1));
%  end 
% %  plot(ovulation_vol)


%% imsegfm3D with img augmentation // augmentation limit the segfm to extend freely
%  ovulation_vol(1:200)=0;
% for j = 1:200
% for i= 279
% I=imread(strcat(fileDirectory,fileName1,num2str(i+(j-1)*500,'%06g'),fileName2));
% BWinv=imcomplement(imbinarize(I,0.2));
% %filter out small pixels 
% BW2 = bwareaopen(BWinv, 60);
% %dilate image to solidate the border 
% se = strel('disk',1);
% BW3 = imdilate(BW2,se);
% %find out the small holes while retain the border 
% filled = imfill (BW3,'holes');
% holes = filled & ~BW3;
% bigholes = bwareaopen(holes, 3000);
% smallholes = holes & ~bigholes;
% BW4 = BW3 | smallholes;
% imshowpair(BW3,BW4,'montage')
% %segment continous region
% BW5=imresize(BW4, [1024 1500]);
% BW5=BW5(270:420,480:720);
% Vol(:,:,i-250)=double(BW5);
% end 
% 
% mask = false(size(Vol)); 
% mask(130,120,29) = true;
% W = graydiffweight(Vol, mask, 'GrayDifferenceCutoff',25);
% % W = gradientweight(Vol,1.5,'RolloffFactor',2,'WeightCutoff',0.8)
% thresh =0.000005;
% BW6 = imsegfmm(W, mask, thresh);
% for i=29
% imwrite(BW6(:,:,i),strcat('F:\code for kohei ovulation project\imsegfm40\imsegfm_Z',num2str(i),'_T',num2str(j),'.tif'));
% end 
%  ovulation_vol(j)=length(find(BW6==1));
% end 
%  plot(ovulation_vol)
%  
 %% imsegfm2d+time // inconsistent between different time

% for j = 1:200
% i= 280;
% I=imread(strcat(fileDirectory,fileName1,num2str(i+(j-1)*500,'%06g'),fileName2));
% Vol(:,:,j)=I(:,:);
% end 
% 
% mask = false(size(Vol)); 
% mask(345,205,151) = true;
% W = graydiffweight(Vol, mask, 'GrayDifferenceCutoff',40);
% thresh = 0.01;
% BW = imsegfmm(W, mask, thresh);
% for i=1:200
% imwrite(BW(:,:,i),strcat('F:\code for kohei ovulation project\imsegfm_time\imsegfm_Z',num2str(i),'.tif'));
% end 

%% k means // not working 
% i= 280;
% I=imread(strcat(fileDirectory,fileName1,num2str(i,'%06g'),fileName2));
% [L,Centers] = imsegkmeans(I,3);
% B = labeloverlay(I,L);
% imshow(B)
% title("Labeled Image")


%% superpixel // NOT WORKING
% A=imread(strcat(fileDirectory,fileName1,num2str(280,'%06g'),fileName2));
% [L,N] = superpixels(A,500);
% figure
% BW = boundarymask(L);
% imshow(imoverlay(A,BW,'cyan'),'InitialMagnification',67)
%% watershed split continous region not working
% A=imread(strcat(fileDirectory,fileName1,num2str(280,'%06g'),fileName2));
% BWinv=imcomplement(imbinarize(A,0.3));
% L = watershed(BWinv);
% L(~BWinv) = 0;
% rgb = label2rgb(L,'jet',[.5 .5 .5]);
% imshow(rgb)
%% image augmentation and imfindcircles \\ the circle range is hard to define getting small later
% mkdir(strcat('F:\code for kohei ovulation project\imfindcircles_crop'));
% for j = 1:200
% A=imread(strcat(fileDirectory,fileName1,num2str(280+(j-1)*500,'%06g'),fileName2));
% BWinv=imcomplement(imbinarize(A,0.2));
% %filter out small pixels 
% BW2 = bwareaopen(BWinv, 60);
% %dilate image to solidate the border 
% se = strel('disk',1);
% BW3 = imdilate(BW2,se);
% %find out the same holes 
% filled = imfill (BW3,'holes');
% holes = filled & ~BW3;
% bigholes = bwareaopen(holes, 3000);
% smallholes = holes & ~bigholes;
% BW4 = BW3 | smallholes;
% %imshowpair(BW3,BW4,'montage')
% %segment continous region
% BW5=imresize(BW4, [1024 1588]);
% BW5=BW5(260:420,500:750);
% 
% 
%         [centers, radii, metric] = imfindcircles(BW5,[30 110],sensitivity=1,Method='twostage',ObjectPolarity='bright');
%         if radii~=0
%   
%             imshow(BW5);
%             viscircles(centers(1:5,:), radii(1:5),'EdgeColor','b');
%             export_fig(strcat('F:\code for kohei ovulation project\imfindcircles_crop\contrast_Z',num2str(j),'.tif'))
%         end 
% end 

%% image augmentation and improp
% mkdir(strcat('F:\code for kohei ovulation project\regionprop'));
% for j = 1:200
% A=imread(strcat(fileDirectory,fileName1,num2str(280+(j-1)*500,'%06g'),fileName2));
% BWinv=imcomplement(imbinarize(A,0.2));
% %filter out small pixels 
% BW2 = bwareaopen(BWinv, 60);
% %dilate image to solidate the border 
% se = strel('disk',1);
% BW3 = imdilate(BW2,se);
% %find out the same holes 
% filled = imfill (BW3,'holes');
% holes = filled & ~BW3;
% bigholes = bwareaopen(holes, 3000);
% smallholes = holes & ~bigholes;
% BW4 = BW3 | smallholes;
% %imshowpair(BW3,BW4,'montage')
% %segment continous region
% BW5=imresize(BW4, [1024 1588]);
% BW5=BW5(260:420,500:750);
% s = regionprops(BW5,'centroid');
% centroids = cat(1,s.Centroid);
% imshow(BW5)
% hold on
% plot(centroids(:,1),centroids(:,2),'b*')
% hold off
% export_fig(strcat('F:\code for kohei ovulation project\regionprop\centroid_Z',num2str(j),'.tif'))
% end 

%% use the y cross section line 
mkdir(strcat('F:\code for kohei ovulation project\imaugmentation'));
diam_cross_t=[];
for j = 1
A=imread(strcat(fileDirectory,fileName1,num2str(280+(j-1)*500,'%06g'),fileName2));
BWinv=imcomplement(imbinarize(A,0.2));
%filter out small pixels 
BW2 = bwareaopen(BWinv, 60);
%dilate image to solidate the border 
se = strel("diamond",1);
BW3 = imdilate(BW2,se);
%find out the same holes 
filled = imfill (BW3,'holes');
holes = filled & ~BW3;
bigholes = bwareaopen(holes, 3000);
smallholes = holes & ~bigholes;
BW4 = BW3 | smallholes;
%imshowpair(BW3,BW4,'montage')
%segment continous region
BW5=imresize(BW4, [1024 1588]);
imshow(BW5);
impixelinfo()
for i = 1:1538
    line= BW5(410,:);
%     if all(line(i:i+50)==0) && line(i+51)==1
%         bar=line(i+51:1588);
        diam=find(diff(find(line==1))~=1);
        diam=diff([0,diam]);
        if find((diam>60)&(diam<300),1)
            diam_cross_t(j)=diam(find((diam>60)&(diam<300),1));
        end 
        
%     end 

% imwrite(BW5,strcat('F:\code for kohei ovulation project\imaugmentation\structure_Z',num2str(j),'.tif'))
end 

% G = findgroups(T.Gender);
% [minHeight,meanHeight,maxHeight] = splitapply(@multiStats,T.Height,G);
end 
plot(diam_cross_t)
%% COMMENTS
%//problem with imfindcircle is that diffrent frame has diffrent size of
%circle that matters a lot 
 
% for j = 1:200
% A=imread(strcat(fileDirectory,fileName1,num2str(280+(j-1)*500,'%06g'),fileName2));
% A=imresize(A, [1024 1588]);
% imwrite(A,strcat('F:\code for kohei ovulation project\imstr\structure_Z',num2str(j),'.tif'))
% end 
% [B,L] = bwboundaries(BW,'noholes');
% imshow(label2rgb(L, @jet, [.5 .5 .5]))
% hold on
% for k = 1:length(B)
%    boundary = B{k};
%    plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2)
% end
