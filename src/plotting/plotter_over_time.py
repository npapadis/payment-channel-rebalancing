from src.plotting.plotter_losses_over_time import plot_fee_losses_over_time
from src.plotting.plotter_fortune_over_time import plot_total_fortune_over_time
from src.plotting.plotter_swaps_over_time import plot_number_of_swaps_over_time

if __name__ == '__main__':

    filename = 'results_104_uniform'

    plot_total_fortune_over_time(filename)
    plot_fee_losses_over_time(filename)
    plot_number_of_swaps_over_time(filename)
