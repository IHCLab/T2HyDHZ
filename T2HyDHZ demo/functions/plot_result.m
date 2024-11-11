function plot_result(Hazy,GT,XDL,time_xdl)
%% Assessment
[ ~ , ~ , sam , uiqi , ~ , mssim , psnr ] = quality_assessment(GT(:,:,:),XDL(:,:,:), 0, 1);
%% Plot
brighten=0.12;

figure('Name','Dehazing Demo');
    
Hazy=plot_rgb(Hazy);
GT=plot_rgb(GT);
XDL=plot_rgb(XDL);

subplot(1,3,1)
imshow(Hazy+brighten);title('Hazy')
xlabel(["PSNR","UIQI","SAM","SSIM",'Time (sec.)']);
subplot(1,3,2)
imshow(GT+brighten);title('GT')
subplot(1,3,3)
imshow(XDL+brighten);title('T^2HyDHZ');
xlabel([round(psnr,3),round(uiqi,3),round(sam,3),round(mssim,3), round((time_xdl),3)]);
end

function xf3=plot_rgb(input)

rgb=[16,11,4];


RGBmax= max(input(:));
RGBmin= min(input(:));

FalseColorf_gray3=input(:,:,[rgb(1),rgb(2),rgb(3)]);
FalseColorf_gray3= (FalseColorf_gray3-RGBmin)/(RGBmax-RGBmin);
xf3=imadjust(FalseColorf_gray3,stretchlim(FalseColorf_gray3),[]);

end