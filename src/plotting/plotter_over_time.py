from pathlib import Path

from src.plotting.plotter_losses_over_time import plot_fee_losses_over_time
from src.plotting.plotter_fortune_over_time import plot_total_fortune_over_time
from src.plotting.plotter_swaps_over_time import plot_number_of_swaps_over_time

if __name__ == '__main__':

    outputs_directory = str(Path("../../outputs").resolve())
    save_at_directory = outputs_directory + "/figures/"

    filename = 'results_test'

    plot_total_fortune_over_time(outputs_directory, save_at_directory, filename)
    plot_fee_losses_over_time(outputs_directory, save_at_directory, filename)
    plot_number_of_swaps_over_time(outputs_directory, save_at_directory, filename)
