%================================================================================
% Input:
% Img_hazy is the hazy HSI for analysis, whose dimension is height*weight*band.
%--------------------------------------------------------------------------------
% Output:
% XDL is the dehazing result of T2HyDHZ, whose dimension is height*weight*band.
% time is the computation time (in seconds).
%================================================================================
clear all;close all;
addpath(genpath('functions'));
addpath(genpath('testing_code'));
%% load data
load Img_hazy.mat 
%% Algorithm
system(['activate dehazing & python testing_code/test.py'])%activate with your own env (i.e., dehazing)
%% Plot
load('results/XDL.mat');
XDL = double(XDL);
plot_result(hazy_sim,X3D,XDL,time_dl);