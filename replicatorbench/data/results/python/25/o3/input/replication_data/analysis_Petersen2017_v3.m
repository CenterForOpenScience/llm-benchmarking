
% This an analysis routine for replication of Petersen et al., 2017
% Experiment 1.
% note that specific details of filepaths, filenames, and definition of
% indices will be updated as project details and the format of experiment
% log files are finalized

function [subjdata] = analysis_Petersen2017_v3(subj)
filelist0={'Example1_Day1',...
    'Example2_Day1','Example2_Day2',...
    'Example3_Day1','Example3_Day2',...
    'Example4_Day1'}';
allfiles0=char(filelist0);
filelist=cellstr(allfiles0(:,1:8));
if nargin < 1
    subjlist=unique(filelist);
else
    filelist=filelist(strcmp(filelist,subj));
    subjlist={subj};
end

datadir=([pwd filesep 'data' filesep]);

for isubj=1:length(subjlist)
    subj=subjlist{isubj};
    Ndays=sum(strcmp(filelist,subj));
    for iday=1:Ndays
        [num,txt]=xlsread([datadir subj '_Day' num2str(iday) '-VAT_replication_scenario.xlsx']);
        txt=txt(6:end,:);

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
                CTOA(irow)=str2double(txt{irow-1,4}(end-2:end));
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
            daydata(iday).prob_Cue(d)=sum(trial_answers((trial_exposures==durations(d)) & trial_sounds==1))/sum(trial_exposures==durations(d) & (trial_sounds==1));
            daydata(iday).prob_NoCue(d)=sum(trial_answers((trial_exposures==durations(d)) & trial_sounds==0))/sum(trial_exposures==durations(d) & (trial_sounds==0));
        end

        
    end
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
    xlabel('Exposure Duration (ms)')
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

Cue_v=cat(1,subjdata.v_Cue);
NoCue_v=cat(1,subjdata.v_NoCue);
figure;hold on
bar(1:2,[mean(NoCue_v),mean(Cue_v)],.5)
errorbar(1:2,[mean(NoCue_v),mean(Cue_v)],[std(NoCue_v),std(Cue_v)]/sqrt(length(subjlist)-1),'k.')
set(gca,'fontsize',14,'xtick',1:2,'xticklabel',{'NoCue','Cue'})
ylabel('Processing Speed, v (letters/s)')


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

