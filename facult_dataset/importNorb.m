% fid=fopen('smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat','r');
% fread(fid,4,'uchar'); % result = [85 76 61 30], byte matrix(in base 16: [55 4C 3D 1E])
% fread(fid,4,'uchar'); % result = [4 0 0 0], ndim = 4
% fread(fid,4,'uchar'); % result = [236 94 0 0], dim0 = 24300 (=94*256+236)
% fread(fid,4,'uchar'); % result = [2 0 0 0], dim1 = 2
% fread(fid,4,'uchar'); % result = [96 0 0 0], dim2 = 96
% fread(fid,4,'uchar'); % result = [96 0 0 0], dim3 = 96
% 
% nb_picture=24300;
% testing_set_left=zeros(24*24,nb_picture);
% testing_set_right=zeros(24*24,nb_picture);
% for i=1:nb_picture
%     if (mod(i,200)==0)
%         i
%     end
%     img=reshape(fread(fid,96*96),96,96);
%     [c,r ] = size(img); % Récupération des 2 dimensions de l'image
%     [ci,ri] = meshgrid(1:4:r, 1:4:c); % Génération de la grille d'interpolation
%     img = interp2(img,ci,ri); % Interpolation des valeurs des pixels
%     testing_set_left(:,i)=img(:);
%     
%     img=reshape(fread(fid,96*96),96,96);
%     [c,r ] = size(img); % Récupération des 2 dimensions de l'image
%     [ci,ri] = meshgrid(1:4:r, 1:4:c); % Génération de la grille d'interpolation
%     img = interp2(img,ci,ri); % Interpolation des valeurs des pixels
%     testing_set_right(:,i)=img(:);
% end
% save('testing_set_left','testing_set_left');
% save('testing_set_right','testing_set_right');

%% category
fid=fopen('smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat','r');
fread(fid,4,'uchar'); % result = [84 76 61 30], int matrix (54 4C 3D 1E)
fread(fid,4,'uchar'); % result = [1 0 0 0], ndim = 1
fread(fid,4,'uchar'); % result = [236 94 0 0], dim0 = 24300
fread(fid,4,'uchar'); % result = [1 0 0 0] (ignore this integer)
fread(fid,4,'uchar'); % result = [1 0 0 0] (ignore this integer)

nb_picture=24300;
testing_cat=zeros(1,nb_picture);
for i=1:nb_picture
    i
    testing_cat(i)=fread(fid,1,'int');
end

save('testing_cat','testing_cat');