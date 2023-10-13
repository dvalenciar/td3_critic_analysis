
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




#folder_results_path = '/home/david_lab/UoA_Repository/td3_critic_analysis/data_results/TD3/walker_walk/'

#curve_data_name ='/seed_10/walker_walk_TD3_Ensemble_size_2_date_10_11_10_38_seed_10_evaluation'
#curve_data_name ='/seed_25/walker_walk_TD3_Ensemble_size_2_date_10_11_10_38_seed_25_evaluation'
#curve_data_name ='/seed_35/walker_walk_TD3_Ensemble_size_2_date_10_11_11_05_seed_35_evaluation'
#curve_data_name ='/seed_45/walker_walk_TD3_Ensemble_size_2_date_10_11_11_05_seed_45_evaluation'
#curve_data_name ='/seed_55/walker_walk_TD3_Ensemble_size_2_date_10_11_10_38_seed_55_evaluation'


folder_results_path = '/home/david_lab/UoA_Repository/td3_critic_analysis/data_results/STD3_Ensemble_2/walker_walk/'
#curve_data_name ='/seed_10/walker_walk_STC_TD3_Ensemble_size_2_date_10_11_13_34_seed_10_evaluation'
#curve_data_name ='/seed_25/walker_walk_STC_TD3_Ensemble_size_2_date_10_11_13_34_seed_25_evaluation'
#curve_data_name ='/seed_25/walker_walk_STC_TD3_Ensemble_size_2_date_10_11_13_34_seed_25_evaluation'
curve_data_name ='/seed_45/walker_walk_STC_TD3_Ensemble_size_2_date_10_11_13_45_seed_45_evaluation'



data_1 = pd.read_csv(folder_results_path+curve_data_name)
plot_function(data_1)

