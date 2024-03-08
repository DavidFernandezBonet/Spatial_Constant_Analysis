import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from scipy.optimize import curve_fit
import scipy.stats
class CurveFitting:
    """
    Usage:
    # Input x and y data
    curve_fitting_object = CurveFitting(df_original['num_nodes'].values, df_original['mean_shortest_path'].values)
    # Select curve to fit
    func_fit = curve_fitting_object.power_model
    # Perform the fit
    curve_fitting_object.perform_curve_fitting(model_func=func_fit)
    # Plot
    curve_fitting_object.plot_fit_with_uncertainty(self, model_func, xlabel, ylabel, title, save_path)

    """
    def __init__(self, x_data, y_data, y_error_std=None, verbose=False):
        """
        x_data: numpy array
        y_data: numpy array
        # TODO: check that this is true
        """
        self.x_data = x_data
        self.y_data = y_data
        self.popt = None
        self.pcov = None
        self.sigma = None
        self.fitError = None
        self.sorted_x = None
        self.sorted_y = None
        self.y_error_std = y_error_std
        self.sorted_y_errors = None  # TODO: this is not well implemented
        self.reduced_chi_squared = None
        self.r_squared = None

        self.fixed_a = None
        self.fixed_b = None

        self.verbose = verbose

    def neg_exponential_model(self, x, a, b, c):
        return a * np.exp(-b * x) + c

    def linear_model(self, x, a, b):
        return a*x + b
    def quadratic_model(self,x, a, b, c):
        return a*x**2 + b*x + c

    def cubic_model(self,x, a, b, c, d):
        return a*x**3 + b*x**2 + c*x + d

    def power_model(self, x, a, b):
        return a * np.power(x, b)

    def power_model_fixed(self, x, b):
        return self.fixed_a * np.power(x, b)

    def power_model_fixed_exp(self, x, a):
        return a * np.power(x, self.fixed_b)
    def power_model_2d_Sconstant(self, x, a):
        # s = 1.10
        # s = 0.51
        s = 0.36
        return s * np.power(x, a)
    def power_model_2d_bi_Sconstant(self, x, a):
        s = 0.9
        return s * np.power(x, a)
    def power_model_3d_Sconstant(self, x, a):
        s = 1.4
        return s * np.power(x, a)
    def power_model_3d_bi_Sconstant(self, x, a):
        s = 1.3
        return s * np.power(x, a)

    def power_model_w_constant(self, x, a, b, c):
        return a * np.power(x, b) + c
    def logarithmic_model(self, x, a, b, c):
        return a * np.log(b * x) + c

    def spatial_constant_dim2(self, x, a, b):
        return a * np.power(x, 1/2) + b

    def spatial_constant_dim2_linearterm(self, x, a):
        return a * np.power(x, 1/2)

    def spatial_constant_dim3(self, x, a, b):
        return a * np.power(x, 1/3) + b

    def spatial_constant_dim3_linearterm(self, x, a):
        return a * np.power(x, 1/3)

    def small_world_model(self, x, a, b):
        return a * np.log(x) + b

    def get_equation_string(self, model_func, inverse_display=False):
        if model_func == self.neg_exponential_model:
            a, b, c = self.popt
            return f'$y = {a:.4f} \exp(-{b:.4f} x) + {c:.4f}$'

        elif model_func == self.linear_model:
            a, b = self.popt
            return f'$y = {a:.4f}x + {b:.4f}$'

        elif model_func == self.quadratic_model:
            a, b, c = self.popt
            return f'$y = {a:.4f}x^2 + {b:.4f}x + {c:.4f}$'

        elif model_func == self.cubic_model:
            a, b, c, d = self.popt
            return f'$y = {a:.4f}x^3 + {b:.4f}x^2 + {c:.4f}x + {d:.4f}$'
        elif model_func == self.power_model:
            a, b = self.popt
            if inverse_display:
                return f'$y = {a:.4f} \cdot x^{{{{1/{1/b:.4f}}}}}$'  ## Works better for dim prediction
            else:
                return f'$y = {a:.4f} \cdot x^{{{b:.4f}}}$'

        elif model_func == self.power_model_fixed:
            b = self.popt
            a = self.fixed_a
            if inverse_display:
                return f'$y = {a} \cdot x^{{{{1/{1/b:.4f}}}}}$'  ## Works better for dim prediction
            else:
                return f'$y = {a} \cdot x^{{{b}}}$'

        elif model_func == self.power_model_fixed_exp:
            a = self.popt
            b = self.fixed_b
            if inverse_display:
                return f'$y = {a} \cdot x^{{{{1/{1/b:.4f}}}}}$'  ## Works better for dim prediction
            else:
                return f'$y = {a} \cdot x^{{{b}}}$'

        elif model_func == self.power_model_2d_Sconstant:
            b = self.popt[0]
            return f'$y = 1.10 \cdot x^{{{b:.4f}}}$'

        elif model_func == self.power_model_3d_Sconstant:
            b = self.popt[0]
            return f'$y = 1.4 \cdot x^{{{b:.4f}}}$'

        elif model_func == self.power_model_2d_bi_Sconstant:
            b = self.popt[0]
            return f'$y = 0.9 \cdot x^{{{b:.4f}}}$'

        elif model_func == self.power_model_3d_bi_Sconstant:
            b = self.popt[0]
            return f'$y = 1.16 \cdot x^{{{b:.4f}}}$'


        elif model_func == self.power_model_w_constant:
            a, b, c = self.popt
            return f'$y = {a:.4f} \cdot x^{{{b:.4f}}} + {c: .4f}$'
        elif model_func == self.logarithmic_model:
            a, b, c = self.popt
            return f'$y = {a:.4f} \cdot \log({b:.4f} x) + {c:.4f}$'

        elif model_func == self.spatial_constant_dim2:
            a, b = self.popt
            return f'$y = {a:.4f} \cdot x^{{1/2}} + {b:.4f}$'

        elif model_func == self.spatial_constant_dim2_linearterm:
            a = self.popt[0]

            return f'$y = {a:.4f} \cdot x^{{1/2}} $'

        elif model_func == self.spatial_constant_dim3:
            a, b = self.popt
            return f'$y = {a:.4f} \cdot x^{{1/3}} + {b:.4f}$'

        elif model_func == self.spatial_constant_dim3_linearterm:
            a = self.popt[0]
            return f'$y = {a:.4f} \cdot x^{{1/3}} $'

        elif model_func == self.small_world_model:
            a, b = self.popt
            return f'$y = {a:.4f} \cdot \log(x) + {b:.4f}$'
        else:
            return 'Unknown model'

    def calculate_durbin_watson(self, residuals):
        diff_res = np.diff(residuals)  # Compute the difference of successive residuals
        sum_diff_res = np.sum(diff_res ** 2)  # Sum of the squared differences
        sum_res = np.sum(residuals ** 2)  # Sum of the squared residuals
        return sum_diff_res / sum_res if sum_res > 0 else 0  # Handle division by zero

    def perform_curve_fitting(self, model_func, p0=None, constant_error=None):
        # Sort the x values while keeping y values matched

        sorted_indices = np.argsort(self.x_data)

        self.sorted_x = self.x_data[sorted_indices]
        self.sorted_y = self.y_data[sorted_indices]
        self.sorted_y_errors = np.full_like(self.y_data, constant_error if constant_error is not None else 1.0)  #TODO: careful with this! Only if we don't have errors

        # Perform curve fitting
        self.popt, self.pcov = curve_fit(model_func, self.sorted_x, self.sorted_y, sigma=self.y_error_std, p0=p0)
        self.sigma = np.sqrt(np.diag(self.pcov))

        # Calculate standard deviation of fit values
        param_combinations = list(product(*[(1, -1)]*len(self.sigma)))
        values = np.array([model_func(self.sorted_x, *(self.popt + np.array(comb) * self.sigma)) for comb in param_combinations])
        self.fitError = np.std(values, axis=0)


        # Calculate residuals and reduced chi-squared  #TODO: not working for now, Is there a meaningful way to associate errors?
        y_fit = model_func(self.sorted_x, *self.popt)
        residuals = self.sorted_y - y_fit

        if self.y_error_std is not None:
            # chi_squared = np.sum((residuals / self.sorted_y_errors) ** 2)
            print("YSTD", self.y_error_std)
            chi_squared = np.sum((residuals / self.y_error_std) ** 2)   # Here I am using the std errors...
            degrees_of_freedom = len(self.sorted_y) - len(self.popt)
            self.reduced_chi_squared = chi_squared / degrees_of_freedom
            print("chi squared", self.reduced_chi_squared)

        # For R squared
        mean_y = np.mean(self.sorted_y)
        sst = np.sum((self.sorted_y - mean_y) ** 2)
        ssr = np.sum(residuals ** 2)
        self.r_squared = 1 - (ssr / sst)

        if self.verbose:
            print("R-squared:", self.r_squared)

        # KS test
        ks_statistic, p_value = scipy.stats.kstest(self.sorted_y, lambda x: model_func(x, *self.popt))
        if self.verbose:
            print("ks stat", ks_statistic, "p-value", p_value)

        # Perform the Anderson-Darling test on the residuals
        ad_result = scipy.stats.anderson(residuals)

        # Store the results
        self.ad_statistic = ad_result.statistic
        self.ad_critical_values = ad_result.critical_values
        self.ad_significance_levels = ad_result.significance_level

        if self.verbose:
            # Output results
            print("Anderson-Darling Statistic:", self.ad_statistic)
            for i in range(len(self.ad_critical_values)):
                sl, cv = self.ad_significance_levels[i], self.ad_critical_values[i]
                print(f"Significance Level {sl}%: Critical Value {cv}")

            print("Covariance error", self.pcov)


            dubson_watson = self.calculate_durbin_watson(residuals=residuals)
            print("Durbin–Watson statistic", dubson_watson)  # values between 1.5 and 2.5 mean no autocorrelation


    def plot_fit_with_uncertainty(self, model_func, xlabel, ylabel, title, save_path):
        plt.close()
        fig, ax = plt.subplots(figsize=(12, 8), dpi=100, facecolor='w', edgecolor='k')
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20
        curveFit = model_func(self.sorted_x, *self.popt)

        # Plot data and fit
        plt.scatter(self.sorted_x, self.sorted_y, label='Data', alpha=0.5, edgecolors='w', zorder=3)
        plt.plot(self.sorted_x, curveFit, linewidth=2.5, color='green', alpha=0.9, label='Fit', zorder=2)

        # Uncertainty areas
        plt.fill_between(self.sorted_x, curveFit - self.fitError, curveFit + self.fitError, color='red', alpha=0.2, label=r'$\pm 1\sigma$ uncertainty')
        plt.fill_between(self.sorted_x, curveFit - 3*self.fitError, curveFit + 3*self.fitError, color='blue', alpha=0.1, label=r'$\pm 3\sigma$ uncertainty')

        # Equation and reduced chi-squared annotation
        equation = self.get_equation_string(model_func)
        r_squared_text = f'R2: {self.r_squared:.2f}'
        annotation_text = f"{equation}\n{r_squared_text}"
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.7, 0.05, annotation_text, fontsize=12, bbox=props, transform=ax.transAxes, verticalalignment='top')

        # Labels and title
        plt.xlabel(xlabel, fontsize=24)
        plt.ylabel(ylabel, fontsize=24)
        plt.title(title, fontsize=28, color='k')
        ax.legend(fontsize=18, loc='best')

        # Save the plot
        plt.savefig(save_path)

    def plot_fit_with_uncertainty_for_dataset(self, x, y, model_func, ax, label_prefix, color, y_position):
        # Perform curve fitting
        self.perform_curve_fitting(model_func)

        # Plot data and fit
        ax.scatter(x, y, label=f'{label_prefix} Data', alpha=0.5, edgecolors='w', zorder=3, color=color)
        curve_fit = model_func(x, *self.popt)
        ax.plot(x, curve_fit, label=f'Fit {label_prefix}', linestyle='--', color=color, zorder=2)

        # Uncertainty areas
        ax.fill_between(x, curve_fit - self.fitError, curve_fit + self.fitError, alpha=0.2, color=color, label=f'Uncertainty {label_prefix}')

        # Equation annotation
        equation = self.get_equation_string(model_func)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.7, y_position, equation, fontsize=12, bbox=props, transform=ax.transAxes, verticalalignment='top')