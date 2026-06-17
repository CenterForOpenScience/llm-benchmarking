
% This an analysis routine for replication of Petersen et al., 2017
% Experiment 1.
% note that specific details of filepaths, filenames, and definition of
% indices will be updated as project details and the format of experiment
% log files are finalized

function [subjdata] = analysis_Petersen2017_v6
filelist0={'1901_Day1_10-24-2020','1901_Day2_11-22-2020'...
    '1902_Day1_11-3-2020','1902_Day2_11-12-2020',...            %'1903_Day1_11-11-2020','1903_Day2_11-17-2020',... <0 perceptual threshold
    '1904_Day1_11-18-2020','1904_Day2_11-18-2020',...           % problems reading in MATLAB **Need to see how to fix it**
    '1905_Day1_11-19-2020','1905_Day2_11-19-2020',...
    '1906_Day1_11-20-2020','1906_Day2_11-20-2020',...
    '1907_Day1_11-23-2020','1907_Day2_11-24-2020',...
    '1908_Day2_11-24-2020',...                                 %'1908_Day1_11-24-2020' didn't have headphones on
    '1910_Day1_12-07-2020','1910_Day2_12-07-2020',...
    '1911_Day1_12-11-2020','1911_Day2_12-11-2020',...           %'1909_Day1_12-07-2020','1909_Day2_12-07-2020' problems reading  
    '1912_Day1_12-14-2020','1912_Day2_12-14-2020',...
    '1913_Day1_12-16-2020','1913_Day2_12-16-2020',...
    '1914_Day1_12-17-2020','1914_Day2_12-17-2020',...
    '1916_Day1_01-05-2021','1916_Day2_01-05-2021',...               %'1915_Day1_12-21-2020'%'1915_Day2_12-21-2020' near 0 accuracy all throughout
    '1917_Day1_01-05-2021','1917_Day2_01-05-2021',...
    '1918_Day1_01-11-2021','1918_Day2_01-12-2021',...
    '1919_Day1_01-13-2021','1919_Day2_01-13-2021',...
    '1920_Day1_01-14-2021','1920_Day2_01-14-2021',...
    '1921_Day1_01-19-2021','1921_Day2_01-22-2021',...           %'1922_Day1_01-19-2021','1922_Day2_01-19-2021' problems reading
    '1923_Day1_01-20-2021','1923_Day2_01-20-2021',...
    '1924_Day1_01-28-2021','1924_Day2_01-28-2021',...
    '1925_Day1_01-29-2021','1925_Day2_01-29-2021',...
    '1926_Day1_01-29-2021','1926_Day2_01-29-2021',...
    '1927_Day1_02-05-2021','1927_Day2_02-05-2021',...
    '1928_Day1_02-05-2021','1928_Day2_02-05-2021',...
    '1929_Day1_02-08-2021','1929_Day2_02-08-2021',...           %'1930_Day1_02-15-2021','1930_Day2_02-15-2021' problems reading (ITI)
    '1931_Day1_02-28-2021','1931_Day2_02-28-2021'}';

temp=char(filelist0);
subjlist=unique(str2num(temp(:,1:4)));
datadir=([pwd filesep 'data' filesep]);

for isubj=1:length(subjlist)
    subj=num2str(subjlist(isubj));
    % get files for day 1 and day 2 for this subj
    day1_filename=dir([datadir subj '_Day1_*.xlsx']);
    day2_filename=dir([datadir subj '_Day2_*.xlsx']);
    
    if isempty(day1_filename)
        day1_txt=[];
        day1_num=[];
    else
        [day1_num,txt]=xlsread([datadir day1_filename.name]);
        day1_txt=txt(6:end,:);
    end
    if isempty(day2_filename)
        day2_txt=[];
        day2_num=[];
    else
        [day2_num,txt]=xlsread([datadir day2_filename.name]);
        day2_txt=txt(6:end,:);
    end
    txt=[day1_txt;day2_txt];
    num=[day1_num;day2_num];

        for irow=1:size(txt,1)
            if isempty(strfind(txt{irow,4},'target: '))
                trial_index(irow)=0;
                exposure_duration(irow)=nan;
                answer_value(irow)=nan;
                sound_cue(irow)=nan;
                CTOA(irow)=nan;
                ITI(irow)=nan;
                target_value(irow)=nan;
                response_value(irow)=nan;
            else
                trial_index(irow)=1;
                workingstring=txt{irow,4};
                exposure_duration(irow)=str2double(workingstring((end-2):end));
                answer_value(irow)=strcmp(txt{irow+2,4},'correct answer');
                sound_cue(irow)=strcmp(txt{irow-2,3},'Sound');
                if ~isempty(txt{irow-1,4})
                    CTOA(irow)=str2double(txt{irow-1,4}(end-2:end));
                else            % This first popped up when there was a stray button press/response
                    CTOA(irow)=str2double(txt{irow-2,4}(end-2:end));
                end
                if isempty(txt{irow-3,4})
                    ITI(irow)=str2double(txt{irow-4,4}(end-3:end));
                else
                    ITI(irow)=str2double(txt{irow-3,4}(end-3:end));
                end
                target_value(irow)=str2double(workingstring(min(strfind(workingstring,':'))+2:min(strfind(workingstring,';'))-1));
                response_value(irow)=num(irow+1,3);
            end
        end
        findex=find(trial_index);
        trial_exposures=exposure_duration(findex);
        trial_answers=answer_value(findex);
        trial_sounds=sound_cue(findex);
        CTOAs=CTOA(findex); % to check the geometric distribution
        ITIs=ITI(findex); % to check the geometric distribution
        targets=target_value(findex); % to double check the answers
        responses=response_value(findex); % to double check the answers

        durations=unique(trial_exposures);
        for d=1:length(durations)
%             subjdata(isubj).prob_Cue(d)=sum(trial_answers((trial_exposures==durations(d)) & trial_sounds==1))/sum(trial_exposures==durations(d) & (trial_sounds==1));
%             subjdata(isubj).prob_NoCue(d)=sum(trial_answers((trial_exposures==durations(d)) & trial_sounds==0))/sum(trial_exposures==durations(d) & (trial_sounds==0));
            
%             daydata(iday).prob_Cue(d)=sum(trial_answers((trial_exposures==durations(d)) & trial_sounds==1))/sum(trial_exposures==durations(d) & (trial_sounds==1));
%             daydata(iday).prob_NoCue(d)=sum(trial_answers((trial_exposures==durations(d)) & trial_sounds==0))/sum(trial_exposures==durations(d) & (trial_sounds==0));
            
            daydata.prob_Cue(d)=sum(trial_answers((trial_exposures==durations(d)) & trial_sounds==1))/sum(trial_exposures==durations(d) & (trial_sounds==1));
            daydata.prob_NoCue(d)=sum(trial_answers((trial_exposures==durations(d)) & trial_sounds==0))/sum(trial_exposures==durations(d) & (trial_sounds==0));
        end

        
%     end
    % combines data from Day1 and Day2 (equal # trials in each so fair to combine
    subjdata(isubj).prob_Cue=mean(cat(1,daydata.prob_Cue),1);
    subjdata(isubj).prob_NoCue=mean(cat(1,daydata.prob_NoCue),1);
    
    tdata=durations/1000;
    [subjdata(isubj).v_Cue,subjdata(isubj).t0_Cue,subjdata(isubj).pg_Cue] = curve_fit_NelderMead(tdata,subjdata(isubj).prob_Cue);
    [subjdata(isubj).v_NoCue,subjdata(isubj).t0_NoCue,subjdata(isubj).pg_NoCue] = curve_fit_NelderMead(tdata,subjdata(isubj).prob_NoCue);
    
    clear trial_index exposure_duration answer_value sound_cue CTOA ITI target_value Response_value
end

%% plotting

figure;hold on
for isubj=1:length(subjlist)
    p_durs=(0:1:80)/1000;
    v_Cue=mean(subjdata(isubj).v_Cue);
    t0_Cue=subjdata(isubj).t0_Cue;
    pg_Cue=subjdata(isubj).pg_Cue;
    
    v_NoCue=mean(subjdata(isubj).v_NoCue);
    t0_NoCue=subjdata(isubj).t0_NoCue;
    pg_NoCue=subjdata(isubj).pg_NoCue;
    
    % use best-fit coefficients to plot model results
    values_Cue=1-exp(-v_Cue*(p_durs-t0_Cue)) + exp(-v_Cue*(p_durs-t0_Cue))*pg_Cue*1/20;
    values_NoCue=1-exp(-v_NoCue*(p_durs-t0_NoCue)) + exp(-v_NoCue*(p_durs-t0_NoCue))*pg_NoCue*1/20;
    values_Cue(p_durs<t0_Cue)=0;
    values_NoCue(p_durs<t0_NoCue)=0;

    plot(p_durs,values_Cue,'r')
    plot(p_durs,values_NoCue,'b')
    plot(durations/1000,mean(subjdata(isubj).prob_Cue,1),'ro')
    plot(durations/1000,mean(subjdata(isubj).prob_NoCue,1),'bo')
    
    xlim([0 .090])
    ylim([0 1])
%     set(gca,'fontsize',14,'xtick',10:10:80)
    xlabel('Exposure Duration (s)')
    ylabel('Prob Correct')
    set(gca,'fontsize',14)
    
    for d=1:length(durations)
        subjdata(isubj).modeledfit_Cue(d)=values_Cue(durations(d)/1000==p_durs);
        subjdata(isubj).modeledfit_NoCue(d)=values_NoCue(durations(d)/1000==p_durs);
    end
    [r]=corrcoef(subjdata(isubj).prob_Cue,subjdata(isubj).modeledfit_Cue);
    subjdata(isubj).VarExplained_Cue=r(1,2)^2;
    [r]=corrcoef(subjdata(isubj).prob_NoCue,subjdata(isubj).modeledfit_NoCue);
    subjdata(isubj).VarExplained_NoCue=r(1,2)^2;
end
legend('Sound Cue','No Cue','location','southeast')

% data notes
% - subj 1915 essentially got every single trial incorrect

Cue_v=cat(1,subjdata.v_Cue);
NoCue_v=cat(1,subjdata.v_NoCue);
Cue_t0=1000*cat(1,subjdata.t0_Cue);
NoCue_t0=1000*cat(1,subjdata.t0_NoCue);

pop1=find(NoCue_t0< 0);
pop2=find(Cue_t0< 0);
unexpected_pop=unique([pop1;pop2]);
subjindex=ones(size(subjdata));
% subjindex(unexpected_pop)=0;

figure;hold on
plot(durations/1000,cat(1,subjdata.prob_Cue),'ro')
plot(durations/1000,cat(1,subjdata.prob_NoCue),'bo')
title('all raw data')
xlabel('Exposure Duration (ms)')
ylabel('Probability of correct')
xlim([0 .1])

% 
% plot(p_durs,values_Cue,'r')
% plot(p_durs,values_NoCue,'b')

Cue_v = Cue_v(logical(subjindex));
NoCue_v = NoCue_v(logical(subjindex));
Cue_t0 = (Cue_t0(logical(subjindex)));
NoCue_t0 = (NoCue_t0(logical(subjindex)));

figure;hold on
bar(1:2,[mean(NoCue_v),mean(Cue_v)],.5)
errorbar(1:2,[mean(NoCue_v),mean(Cue_v)],[std(NoCue_v),std(Cue_v)]/sqrt(length(subjlist)-1),'k.')
set(gca,'fontsize',14,'xtick',1:2,'xticklabel',{'NoCue','Cue'})
ylabel('Processing Speed, v (letters/ms)')
[h,p,~,tstats]=ttest(NoCue_v,Cue_v);
title(['Velocity: t(' num2str(tstats.df) ') = ' num2str(abs(tstats.tstat)) ', p = ' num2str(p)])


figure;hold on
bar(1:2,[mean(NoCue_t0),mean(Cue_t0)],.5)
errorbar(1:2,[mean(NoCue_t0),mean(Cue_t0)],[std(NoCue_t0),std(Cue_t0)]/sqrt(length(subjlist)-1),'k.')
set(gca,'fontsize',14,'xtick',1:2,'xticklabel',{'NoCue','Cue'})
ylabel('Perceptual threshold, t0 (ms)')
[h,p,~,tstats]=ttest(NoCue_t0,Cue_t0);
title(['Threshold: t(' num2str(tstats.df) ') = ' num2str(abs(tstats.tstat)) ', p = ' num2str(p)])

figure
subplot(1,2,1)
hist(NoCue_t0)
xlabel('t0 (threshold)')
ylabel('count')
title('NoCue')

subplot(1,2,2)
hist(Cue_t0)
xlabel('t0 (threshold)')
ylabel('count')
title('Cue')


%% Nelder-Mead curve fitting subroutine
function [output_v,output_t0,output_pg] = curve_fit_NelderMead(tdata,ydata)

fun=@(x) sum((ydata - ((1-exp(-x(1)*(tdata-x(2)))) + (exp(-x(1)*(tdata-x(2))))*x(3)*1/20)).^2);

v_init=[20,50,100];
t0_init=[0,.025,.05];
pg_init=[0,.5,1];

x0=[v_init;t0_init;pg_init];
[bestx]=fminsearch(fun,x0);

output_v=bestx(1);
output_t0=bestx(2);
output_pg=bestx(3);

