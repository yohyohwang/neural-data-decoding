%% Yohyoh Wang
%% Lab 6 

%% Population vector decoder

%% Train
clear all; close all; load('Lab5_CenterOutTrain.mat');
load('sT_p.mat'); %preferred direction for each neuron
rad=0:pi/4:2*pi;
spikeWindow=2;

%get means
for n=1:length(unit)
    nUse=unit(n).times;
    for dir=1:max(direction)
        usego=go(direction==dir); %only use go times for this direction
        sT_all=[];
        for goTime=1:length(usego)
            spikeTimes=nUse(nUse >= usego(goTime)-spikeWindow/2 & nUse <= usego(goTime)+spikeWindow/2); %limit spike times to each epoch
            spikeNum=length(spikeTimes);
            sT_all=[sT_all;spikeNum];
        end
        sT_means(n,dir)=mean(sT_all)/spikeWindow;
    end
end

%fit cosine function
cos_fun=@(p,theta) p(1)+p(2)*cos(theta-p(3)); %Cosine function
for n=1:size(sT_means,1)
    sT_fit(n,:) = cos_fun(p(n,:),rad);
end

%preferred directions
if exist('sT_pref','var')
else
    [m sT_pref] = max(sT_fit'); %tuning curve preferred directions
end

[m n_pref] = max(sT_means');
tuningAcc=sum(sT_pref==n_pref)/length(sT_pref);
sT_pref=rad(sT_pref);

%% Test
load('Lab6_CenterOutTest.mat')
spikeWindow=1;

for dir=1:max(direction)
    rad(dir)=(dir-1)*(pi/4);
    degrees(dir)=(dir-1)*45;
end

popVec=zeros(length(go),2);
for trial=1:length(go)
    for neuron=1:length(unit)
        sR=sum(unit(neuron).times >= go(trial)-spikeWindow & unit(neuron).times <= go(trial))/spikeWindow; %spike rate
        popVec(trial,1)=popVec(trial,1)+sR*cos(sT_pref(neuron)); % weight by firing rate: x
        popVec(trial,2)=popVec(trial,2)+sR*sin(sT_pref(neuron)); % y
    end
    popDir(trial,1)=atan2(popVec(trial,2),popVec(trial,1));
end

popDir(popDir<0)=popDir(popDir<0)+2*pi;
for trial=1:length(popDir)
    [m popDir(trial,2)]=min(abs(popDir(trial,1)-rad));
end

%% Q1 Performance
popAcc=sum(popDir(:,2)==direction)/length(popDir);

disp(['The population vector was accurate on ' num2str(popAcc*100) '% of trials.'])

%% Q2

disp(['The cosine tuning curve correctly described ' num2str(tuningAcc*100) '% of tuning preferences.'])

hist(n_pref,8); title(['Distribution of preferred direction'])
ylabel(['Frequency']); xlabel(['Direction']); 

%% Q2 Additional comments

% Assumption 1: neurons are cosine tuned
% The cosine tuning function correctly described the peak of firing for 44%
% of the neurons in the training set. This is an above-chance performance
% (chance performance would be 1/8, or 12.5%), but there may be better
% models for describing tuning that could account for more neurons'
% responses.

% Assumption 2: preferred directions are uniformly distributed.
% The histogram of preferred directions shows that there is a mostly
% uniform distribution (around 15-20 neurons) for tuning in most
% directions, based off of the direction which elicited the maximum mean
% firing rate over all trials.

%% Q3 Weighting with change in firing rate from baseline 

for neuron=1:length(unit)
    for trial=1:length(go)
        sR(neuron,trial)=sum(unit(neuron).times >= go(trial)-spikeWindow & unit(neuron).times <= go(trial))/spikeWindow; %spike rate
    end
    baseline(neuron)=mean(sR(neuron,:));
    sR_baseline(neuron,:)=(sR(neuron,:)-baseline(neuron))/baseline(neuron);
end

popVec_b=zeros(length(go),2);
for trial=1:length(go)
    for neuron=1:length(unit)
        popVec_b(trial,1)=popVec_b(trial,1)+sR_baseline(neuron,trial)*cos(sT_pref(neuron)); % weight by firing rate: x
        popVec_b(trial,2)=popVec_b(trial,2)+sR_baseline(neuron,trial)*sin(sT_pref(neuron)); % y
    end
    popDir(trial,1)=atan2(popVec_b(trial,2),popVec_b(trial,1));
end

popDir(popDir<0)=popDir(popDir<0)+2*pi;
for trial=1:length(popDir)
    [m popDir(trial,2)]=min(abs(popDir(trial,1)-rad));
end

popAcc=sum(popDir(:,2)==direction)/length(popDir);

disp(['The population vector with baselined trials was accurate on ' num2str(popAcc*100) '% of trials.'])

%% Q3 Comments

% The baseline firing rate for each neuron was averaged over all trials,
% then subtracted from the spike count in the 1-s encoding window for each 
% trial and converted into a percentage change from the baseline firing rate. 
% This resulted in both positive and negative deviations from the
% baseline firing rate, and the deviation from the baselined firing rate
% was used instead of the spiking rate. 

% Using the change in baseline firing rate resulted in an incremental
% improvement, from 36.25% to 41.25%. However, the population vector
% decoding method is still quite low, indicating that spike rates are not
% sufficient to provide full information about intended movement direction.

%% Maximum likelihood decoder

%% Train with poisspdf

clear all; close all
load('Lab5_CenterOutTrain.mat');
spikeWindow=1;

%get means
for n=1:length(unit)
    nUse=unit(n).times;
    for dir=1:max(direction)
        usego=go(direction==dir); %only use go times for this direction
        sT_all=[];
        for goTime=1:length(usego)
            spikeTimes=nUse(nUse >= usego(goTime)-spikeWindow & nUse <= usego(goTime)); %limit spike times to each epoch
            spikeNum=length(spikeTimes);
            sT_all=[sT_all;spikeNum];
        end
        sT_means(n,dir)=mean(sT_all)/spikeWindow;
        sT_error(n,dir)=std(sT_all/spikeWindow)/sqrt(length(sT_all));
    end
end

save('sT_means.mat','sT_means');

%% Test
clear all; close all; 
load('Lab6_CenterOutTest.mat'); 
load('sT_means.mat'); %mean firing rate for 

spikeWindow=1;

for dir=1:max(direction)
    dir_prior(dir)=sum(direction==dir)/length(direction);
end

% prd=zeros(length(go),length(unit),max(direction));
for goInd=1:length(go)
    for n=1:length(unit)
        sR=sum(unit(n).times >= go(goInd)-spikeWindow & unit(n).times <= go(goInd)); %spike rate for each trial
        prd(n,:)=log(poisspdf(sR,sT_means(n,:)));
    end
    pRd(goInd,:)=sum(prd,1);
end

[m dir_poiss]=max(pRd');

%% Q1 Performance
poissAcc=sum(dir_poiss'==direction)/length(direction);

disp(['Assuming a Poisson firing rate model, the maximum likelihood algorithm was accurate on ' num2str(poissAcc*100) '% of trials.'])

%% Maximum likelihood performance (Poisson)

% The performance of the maximum likelihood algorithm with a Poisson firing
% rate model was much higher than that of the population vector algorithm. 

%% Train with normpdf

for goInd=1:length(go)
    for n=1:length(unit)
        sR=sum(unit(n).times >= go(goInd)-spikeWindow & unit(n).times <= go(goInd)); %spike rate for each trial
        prd_norm(n,:)=log(normpdf(sR,(sT_means(n,:)),std(sT_means(n,:))));
    end
    pRd_norm(goInd,:)=sum(prd_norm,1);
end

[m dir_norm]=max(pRd_norm');

%% Q4 Performance

normAcc=sum(dir_norm'==direction)/length(direction);
disp(['Assuming a Gaussian firing rate model, the maximum likelihood algorithm was accurate on ' num2str(normAcc*100) '% of trials.'])

%% Maximum likelihood performance (Gaussian) 

% At high firing rates, the distribution of firing rates for a neuron
% approximates a Gaussian distribution. The Gaussian firing rate model gave
% a lower performance (63.75% rather than 68.75%). This model was not as
% accurate at capturing the activity of neurons with lower average spike
% rates. Since a large proportion of the neurons in the training set did
% not have particularly high spiking rates, the Gaussian firing model had a
% lower performance than the Poisson firing model. 

hist(mean(sT_means'),15); title(['Firing rate distribution'])
xlabel(['Mean firing rate (spikes/s)'])
ylabel(['Frequency (# neurons)'])

