
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_function(data_frame):

    data_dict = {'steps': data_frame.step, 'avg_episode_reward': data_frame.avg_episode_reward}
    df_1      = pd.DataFrame(data=data_dict)

    figure, ax = plt.subplots()
    figure.set_figwidth(10)

    sns.lineplot(data=df_1, x="steps", y="avg_episode_reward", label='Reward')

    ax.axhline(y=1000, color="black", linestyle="dashed")  # horizontal Line
    ax.set_xlim(0, 1_000_000)  # axes limit

    ax.grid(linewidth=0.3)
    plt.xlabel("Steps", fontsize=20)
    plt.ylabel("Average Reward", fontsize=20)
    plt.title("Title one", fontsize=20)
    plt.show()




folder_results_path = '/home/david_lab/UoA_Repository/td3_critic_analysis/data_results/'

#curve_data_name ='reacher_hard_TD3_Ensemble_size_2_date_10_09_14_10_evaluation_seed_123'
#curve_data_name ='reacher_hard_STC_TD3_Ensemble_size_2_date_10_09_14_10_evaluation_seed_123'
#curve_data_name ='reacher_hard_STC_TD3_Ensemble_size_5_date_10_09_17_05_evaluation_seed_123'
#curve_data_name ='reacher_hard_STC_TD3_Ensemble_size_10_date_10_09_17_04_evaluation_seed_123'

#curve_data_name ='reacher_hard_TD3_Ensemble_size_2_date_10_10_09_38_seed_753_evaluation'
#curve_data_name ='reacher_hard_TD3_Ensemble_size_2_date_10_09_14_10_evaluation_seed_123'

#curve_data_name ='reacher_hard_STC_TD3_Ensemble_size_2_date_10_10_09_38_seed_753_evaluation'
#curve_data_name ='reacher_hard_STC_TD3_Ensemble_size_2_date_10_10_10_55_seed_753_evaluation'

#curve_data_name ='reacher_hard_STC_TD3_Ensemble_size_5_date_10_10_11_19_seed_753_evaluation'
#curve_data_name ='reacher_hard_STC_TD3_Ensemble_size_5_date_10_10_09_38_seed_753_evaluation'



data_1 = pd.read_csv(folder_results_path+curve_data_name)
plot_function(data_1)

