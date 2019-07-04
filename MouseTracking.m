 /
 function MouseTracking()
    obj = setupSystemObjects();%读取视频，检测移动对象，并显示结果的系统对象obj
    tracks = initializeTracks(); %初始化轨道数组
    nextId = 1; 
    a=1;
    SE=strel('square',50);%用于前期形态学腐蚀使用
    while ~isDone(obj.reader)%使用卡尔曼滤波器跟踪预测目标
        %以下函数在下面均有说明
        frame = readFrame();
        [centroids, bboxes, mask] = detectObjects(frame);
        predictNewLocationsOfTracks();
        [assignments, unassignedTracks, unassignedDetections] = detectionToTrackAssignment();
        updateAssignedTracks();
        updateUnassignedTracks();
        deleteLostTracks();
        createNewTracks();
        [cenb1x,cenb1y,cenb2x,cenb2y]=displayTrackingResults();%显示视频时返回两个小鼠的坐标点并储存，用于后期plot绘制小鼠轨迹
        if a==1
                        cent1x=cenb1x;
                        cent1y=cenb1y;
                        cent2x=cenb2x;
                        cent2y=cenb2y;
                    else
                        cent1x=[cent1x cenb1x];
                        cent1y=[cent1y cenb1y];
                        cent2x=[cent2x cenb2x];
                        cent2y=[cent2y cenb2y];
        end
        a=a+1;
    end
    %绘制小鼠的轨迹图
    plot(cent1x,cent1y);
    hold on;
    plot(cent2x,cent2y,'-r');
    set(gca,'YDir','reverse');
    legend('一号小鼠','二号小鼠',3);
    title('小鼠轨迹图');
    %读取视频
    function obj = setupSystemObjects()
        obj.reader = vision.VideoFileReader('mousevideo.mp4');
        obj.videoPlayer = vision.VideoPlayer();
    end
    %创建轨道数组用于跟踪小鼠目标
    function tracks = initializeTracks()
        tracks = struct(...
            'id', {}, ...%轨道的id,用于标记轨道
            'bbox', {}, ...%小鼠外围的边界框
            'kalmanFilter', {}, ...%卡尔曼滤波器用于预测下一个点的位置
            'age', {}, ...%自首次检测到轨道以来的帧数
            'totalVisibleCount', {}, ...%检测到轨道的帧总数（可见）
            'consecutiveInvisibleCount', {});%未检测到轨道的连续帧数（不可见）。
    end
    %读取视频帧
    function frame = readFrame()
       frame = obj.reader.step();
    end
    %检测前景，获取前景连通区域的中心点和边界框
    function [centroids, bboxes, mask] = detectObjects(frame)
        frame1=im2uint8(frame);
        mask =zeros(1080,1920,3,'uint8');
        for i=100:980
            for j=464:1340
                mask(i,j,:)=frame1(i,j,:);
            end
        end
        mask=im2bw(mask,0.2);
        mask=bwareaopen(mask, 200);
        mask=imfill(mask,'holes');
        mask = imopen(mask, strel('rectangle', [40,40]));
        if a>270&&a<283
            mask=imerode(mask,SE);
        end
        STATS = regionprops(mask,'Centroid','BoundingBox');
        centroids=cat(1,STATS.Centroid);
        bboxes=cat(1,STATS.BoundingBox);
    end
    %使用卡尔曼滤波器预测当前帧中每个轨道的质心，并相应地更新其边界框
    function predictNewLocationsOfTracks()
        for i = 1:length(tracks)%遍历已跟踪的轨迹
            bbox = tracks(i).bbox;
            predictedCentroid = predict(tracks(i).kalmanFilter);%使用卡尔曼滤波器进行预测，得到预测中心
            %移动边界框，使其中心位于预测位置
            predictedCentroid = int32(predictedCentroid) - int32(bbox(3:4) / 2);
            tracks(i).bbox = [predictedCentroid, bbox(3:4)];
        end
    end
    %通过最小化成本来完成将当前帧中的对象检测分配给现有轨道。
    function [assignments, unassignedTracks, unassignedDetections] = detectionToTrackAssignment()
        %获取轨迹和新检测的目标的个数
        nTracks = length(tracks);
        nDetections = size(centroids, 1);
        cost = zeros(nTracks, nDetections);%损失函数矩阵，行代表轨迹，列代表新检测目标
        for i = 1:nTracks
            cost(i, :) = distance(tracks(i).kalmanFilter, centroids);%对于每个轨迹，使用它们的卡尔曼滤波器预测的结果，与每个新检测的目标中心计算欧式距离，存入损失函数矩阵（cost矩阵）
        end
        costOfNonAssignment = 40;%设置阈值，当损失函数矩阵中的某个值小于40时，则不分配轨迹
        [assignments, unassignedTracks, unassignedDetections] = assignDetectionsToTracks(cost, costOfNonAssignment);%使用匈牙利匹配算法根据cost矩阵和阈值分配矩阵并检测目标
    end
    function updateAssignedTracks()%更新已分配的轨迹
        numAssignedTracks = size(assignments, 1);
        for i = 1:numAssignedTracks
            trackIdx = assignments(i, 1);
            detectionIdx = assignments(i, 2);
            centroid = centroids(detectionIdx, :);
            bbox = bboxes(detectionIdx, :);
            correct(tracks(trackIdx).kalmanFilter, centroid);%根据轨迹对应的检测目标位置中心修正它 的卡尔曼滤波器
            tracks(trackIdx).bbox = bbox;
            tracks(trackIdx).age = tracks(trackIdx).age + 1;%轨迹年龄增加1
            tracks(trackIdx).totalVisibleCount = tracks(trackIdx).totalVisibleCount + 1;%轨迹可见帧数增加1
            tracks(trackIdx).consecutiveInvisibleCount = 0;%轨迹连续不可见帧数清零
        end
    end
    function updateUnassignedTracks()%更新未分配轨迹
        for i = 1:length(unassignedTracks)
            ind = unassignedTracks(i);
            tracks(ind).age = tracks(ind).age + 1;%轨迹年龄加1
            tracks(ind).consecutiveInvisibleCount =  tracks(ind).consecutiveInvisibleCount + 1;%轨迹连续不可见帧数加1
        end
    end
    function deleteLostTracks()%删除丢失的轨迹
    if isempty(tracks)
        return;
    end
    invisibleForTooLong = 20;%连续不可见帧数大于等于20时丢弃轨迹
    ageThreshold = 8;%当年龄小于8时，若总可见帧数与年龄的比值小于0.6，则丢弃轨迹；否则再根据连续不可见帧数丢弃轨迹
    ages = [tracks(:).age];
    totalVisibleCounts = [tracks(:).totalVisibleCount];
    visibility = totalVisibleCounts ./ ages;
    lostInds = (ages < ageThreshold & visibility < 0.6) | ...
        [tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;
    tracks = tracks(~lostInds);
    end
    function createNewTracks()%创建新轨迹
    centroids = centroids(unassignedDetections, :);
    bboxes = bboxes(unassignedDetections, :);
    for i = 1:size(centroids, 1)
        centroid = centroids(i,:);
        bbox = bboxes(i, :);
        kalmanFilter = configureKalmanFilter('ConstantAcceleration', centroid, [1 1 1]*1e5,[25,10,10],25);%创建新的二阶加速度卡尔曼滤波器
        newTrack = struct(...
            'id', nextId, ...
            'bbox', bbox, ...
            'kalmanFilter', kalmanFilter, ...
            'age', 1, ...
            'totalVisibleCount', 1, ...
            'consecutiveInvisibleCount', 0);
        tracks(end + 1) = newTrack;
        nextId = nextId + 1;
    end
    end
    function [cen1x,cen1y,cen2x,cen2y]=displayTrackingResults()%显示跟踪结果并返回轨迹中心点
        frame = im2uint8(frame);
        mask = uint8(repmat(mask, [1, 1, 3])) .* 255;
        minVisibleCount = 1;%可见帧数阈值，大于阈值才显示
        if ~isempty(tracks)
            reliableTrackInds = ...
                [tracks(:).totalVisibleCount] > minVisibleCount;
            reliableTracks = tracks(reliableTrackInds);
            if ~isempty(reliableTracks)
                bboxes = cat(1, reliableTracks.bbox);
                ids = int32([reliableTracks(:).id]);
                labels = cellstr(int2str(ids'));
                predictedTrackInds = ...
                    [reliableTracks(:).consecutiveInvisibleCount] > 0;
                isPredicted = cell(size(labels));
                isPredicted(predictedTrackInds) = {' predicted'};
                labels = strcat(labels, isPredicted);
                frame = insertObjectAnnotation(frame, 'rectangle', bboxes, labels,'Color',{'cyan','yellow'},'FontSize',22);
            end
            %获取中心点坐标（有两只小鼠，两个中心点坐标）
            cena1x=bboxes(1,1)+bboxes(1,3)/2;
            cena1y=bboxes(1,2)+bboxes(1,4)/2;
            cena2x=bboxes(2,1)+bboxes(2,3)/2;
            cena2y=bboxes(2,2)+bboxes(2,4)/2;    
            cen1x=cena1x;
            cen1y=cena1y;
            cen2x=cena2x;
            cen2y=cena2y;
        end
        obj.videoPlayer.step(frame);
    end
end
/
