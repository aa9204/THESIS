"""
Writes ontology/signals_relations_base.json: the BASE list of candidate relation types for Signals & Systems
(no classes). The full-book run counts how often each is used; the frequency table then cuts the list to ~40.

Sources: the 56 draft relations (Gemini + ChatGPT), plus relations that the first full run kept inventing
(OTHER:* with several uses), plus common textbook relations for chapters 1-11. LLM-proposed; see notes.
"""
import json, os
R = [
 # --- taxonomy / generic
 ("is_subclass_of","Every X is a kind of Y (class to broader class, e.g. FIR system -> LTI system)."),
 ("is_instance_of","X is a particular example of the class Y."),
 ("is_special_case_of","X is a specific case of the more general Y under extra conditions."),
 ("is_generalization_of","X is the more general form of Y."),
 ("is_equivalent_to","X and Y are the same thing, different names for it, or interchangeable."),
 ("is_defined_as","X is defined to be Y (a definition)."),
 ("is_component_of","X is a part of Y."),
 ("is_decomposed_into","X can be broken into the components Y."),
 ("consists_of","X is built from the elements Y."),
 ("corresponds_to","X is paired with Y across domains or representations, without being identical."),
 ("is_counterpart_of","X is the discrete-time (or continuous-time) analogue of Y."),
 ("is_dual_of","X and Y mirror each other under a duality (time/frequency, analysis/synthesis)."),
 ("is_inverse_of","X undoes Y (inverse system, inverse transform, accumulation/difference)."),
 ("is_analogous_to","X plays a role similar to Y in another setting."),
 ("is_compared_with","X is contrasted with Y."),
 # --- logic / conditions
 ("implies","If X holds then Y follows."),
 ("requires_condition","X holds, exists or is valid only when condition Y is met."),
 ("is_necessary_for","Y cannot hold unless X holds."),
 ("is_sufficient_for","If X holds then Y holds."),
 ("holds_for","Statement or property X is valid for the class or range Y."),
 ("does_not_hold_for","X fails for the class or range Y."),
 ("reduces_to","X becomes the simpler form Y in a special case or after simplification."),
 ("is_bounded_by","X is limited above or below by Y."),
 ("is_approximated_by","X is approximated by Y."),
 # --- properties
 ("has_property","X has the attribute or characteristic Y (causal, linear, stable, zero phase...)."),
 ("lacks_property","X does not have the property Y."),
 ("preserves","Operation or system X leaves the quantity Y unchanged."),
 ("changes","Operation or system X alters the quantity Y."),
 ("is_characterized_by","X is described or determined by Y (e.g. an LTI system by its impulse response)."),
 ("characterized_by","X is described or determined by Y (e.g. an LTI system by its impulse response)."),
 ("exhibits","X shows the behaviour Y (ringing, overshoot, aliasing)."),
 # --- representation
 ("is_represented_by","X can be expressed or written as Y (an expansion, form or representation)."),
 ("is_expressed_in_terms_of","X is written using Y."),
 ("has_fourier_transform","Y is the Fourier transform of X."),
 ("has_laplace_transform","Y is the Laplace transform of X."),
 ("has_z_transform","Y is the z-transform of X."),
 ("has_inverse_transform","Y is the inverse transform of X."),
 ("is_transform_pair_with","X and Y are a transform pair."),
 ("has_fourier_series","Y is the Fourier series of periodic signal X."),
 ("has_fourier_coefficient","Y is a Fourier series coefficient of periodic signal X."),
 # --- math operations
 ("is_derivative_of","X is the derivative of Y."),
 ("is_integral_of","X is the integral of Y."),
 ("is_first_difference_of","X is the first difference of Y."),
 ("is_running_sum_of","X is the running sum (accumulation) of Y."),
 ("is_running_integral_of","X is the running integral of Y."),
 ("is_convolved_with","X is convolved with Y."),
 ("convolution_produces","Convolving X produces Y."),
 ("is_multiplied_by","X is multiplied by Y (time or frequency domain)."),
 ("is_sum_of","X is the sum of Y."),
 ("is_product_of","X is the product of Y."),
 ("is_reciprocal_of","X is the reciprocal of Y."),
 ("is_shifted_in_time_by","X is delayed or advanced by Y."),
 ("is_scaled_by","X is scaled in time or amplitude by Y."),
 ("is_reflected_by","X is time-reversed by Y."),
 ("is_conjugate_of","X is the complex conjugate of Y."),
 ("equals","X is numerically equal to Y."),
 # --- system relations
 ("has_input","Y is an input of system X."),
 ("produces_output","System X produces output Y."),
 ("is_response_to","X is the response of a system to input Y."),
 ("has_impulse_response","Y is the impulse response of system X."),
 ("has_step_response","Y is the step response of system X."),
 ("has_frequency_response","Y is the frequency response of system X."),
 ("has_transfer_function","Y is the system function H(s) or H(z) of system X."),
 ("governed_by_equation","X obeys the differential or difference equation Y."),
 ("has_eigenfunction","Y is an eigenfunction of system or operation X."),
 ("has_eigenvalue","Y is the eigenvalue of X for its eigenfunction."),
 ("is_cascade_of","System X is a series connection of Y."),
 ("is_parallel_combination_of","System X is a parallel connection of Y."),
 ("has_order","System or filter X has order Y."),
 ("has_time_constant","System X has time constant Y."),
 ("has_damping_ratio","System X has damping ratio Y."),
 ("has_natural_frequency","System X has natural frequency Y."),
 ("has_gain","X has gain Y."),
 ("has_group_delay","X has group delay Y."),
 ("has_memory_of","System X depends on past values Y."),
 # --- spectra / Fourier
 ("has_magnitude_spectrum","Y is the magnitude spectrum of X."),
 ("has_phase_spectrum","Y is the phase spectrum of X."),
 ("has_fundamental_frequency","Y is the fundamental frequency of periodic signal X."),
 ("has_harmonic","Y is a harmonic of X."),
 ("satisfies_parseval_relation","X obeys Parseval's relation (energy equal in both domains)."),
 ("converges_to","X (series, sequence, integral) converges to Y."),
 ("has_bandwidth","X has bandwidth Y."),
 ("has_window_function","Y is the window function applied to X."),
 ("exhibits_gibbs_phenomenon","Partial sums of X overshoot near discontinuities."),
 # --- filters
 ("is_filtered_by","Signal X is passed through filter Y."),
 ("has_passband","Filter X passes the band Y."),
 ("has_stopband","Filter X rejects the band Y."),
 ("has_cutoff_frequency","Filter X has cutoff frequency Y."),
 ("attenuates","X reduces the amplitude of Y."),
 ("passes","Filter X lets Y through."),
 # --- Laplace / z
 ("has_region_of_convergence","The transform X converges for the set of values Y (ROC)."),
 ("has_pole","Y is a pole of X."),
 ("has_zero","Y is a zero of X."),
 ("evaluated_via_pole_zero_plot","X is evaluated or sketched from its pole-zero plot Y."),
 ("determines_location_of","X fixes where Y lies (e.g. poles in the s-plane)."),
 # --- sampling
 ("is_sampled_by","Signal X is sampled by process Y."),
 ("produces_samples","Sampling X yields the samples Y."),
 ("requires_sampling_rate","Sampling X requires at least rate Y."),
 ("can_cause_aliasing","Operation X can produce aliasing Y."),
 ("is_reconstructed_by","The sampled signal X is recovered by process Y."),
 ("is_interpolated_by","X is interpolated by method Y."),
 ("is_decimated_by","X is decimated (downsampled) by Y."),
 ("is_upsampled_by","X is upsampled by Y."),
 # --- communication
 ("is_modulated_by","Carrier or signal X is modulated by Y."),
 ("is_demodulated_by","The modulated signal X is recovered by demodulation method Y."),
 ("has_carrier","Modulated signal X uses carrier Y."),
 ("is_multiplexed_with","X shares a channel with Y by multiplexing."),
 ("has_sideband","Modulated signal X has sideband Y."),
 ("has_modulation_index","Modulated signal X has modulation index Y."),
 # --- feedback / stability
 ("affects_stability_of","X determines or influences the stability of Y (includes tests for stability)."),
 ("is_stabilized_by","System X is made stable by Y."),
 ("has_feedback_path","System X has feedback path Y."),
 ("has_open_loop_transfer_function","Y is the open-loop system function of X."),
 ("has_closed_loop_transfer_function","Y is the closed-loop system function of X."),
 ("has_gain_margin","Y is the gain margin of X."),
 ("has_phase_margin","Y is the phase margin of X."),
 # --- usage / derivation
 ("is_used_for","X is used to achieve Y."),
 ("is_used_in_analysis_of","X is a tool for analysing Y."),
 ("is_applied_to","X is applied to Y."),
 ("is_computed_by","X is calculated using method Y."),
 ("has_solution","Y is the solution of equation or problem X."),
 ("is_derived_from","X is obtained from Y."),
 ("is_illustrated_by","X is shown by example or figure Y."),
 ("causes","X produces the effect Y."),
 ("results_in","X leads to the outcome Y."),
]
names=[n for n,_ in R]
dup={n for n in names if names.count(n)>1}
assert not dup, dup
# characterized_by was kept from the first draft; is_characterized_by is a duplicate spelling -> drop the duplicate
R=[(n,d) for n,d in R if n!="is_characterized_by"]
out={"version":"base","relations":[{"name":n,"definition":d} for n,d in R]}
p=os.path.join(os.path.dirname(os.path.abspath(__file__)),"signals_relations_base.json")
json.dump(out,open(p,"w"),indent=2)
print("wrote",len(R),"base relations to",p)