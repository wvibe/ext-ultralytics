# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.utils import LOGGER, SETTINGS, TESTS_RUNNING
from ultralytics.utils.metrics import DetMetrics, OBBMetrics, PoseMetrics, SegmentMetrics
from ultralytics.utils.torch_utils import model_info_for_loggers

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS["wandb"] is True  # verify integration is enabled
    import wandb as wb

    assert hasattr(wb, "__version__")  # verify package is not directory
    _processed_plots = {}

except (ImportError, AssertionError):
    wb = None


def _custom_table(x, y, classes, title="Precision Recall Curve", x_title="Recall", y_title="Precision"):
    """
    Create and log a custom metric visualization to wandb.plot.pr_curve.

    This function crafts a custom metric visualization that mimics the behavior of the default wandb precision-recall
    curve while allowing for enhanced customization. The visual metric is useful for monitoring model performance across
    different classes.

    Args:
        x (list): Values for the x-axis; expected to have length N.
        y (list): Corresponding values for the y-axis; also expected to have length N.
        classes (list): Labels identifying the class of each point; length N.
        title (str, optional): Title for the plot.
        x_title (str, optional): Label for the x-axis.
        y_title (str, optional): Label for the y-axis.

    Returns:
        (wandb.Object): A wandb object suitable for logging, showcasing the crafted metric visualization.
    """
    import pandas  # scope for faster 'import ultralytics'

    df = pandas.DataFrame({"class": classes, "y": y, "x": x}).round(3)
    fields = {"x": "x", "y": "y", "class": "class"}
    string_fields = {"title": title, "x-axis-title": x_title, "y-axis-title": y_title}
    return wb.plot_table(
        "wandb/area-under-curve/v0", wb.Table(dataframe=df), fields=fields, string_fields=string_fields
    )


def _plot_curve(
    x,
    y,
    names=None,
    id="precision-recall",
    title="Precision Recall Curve",
    x_title="Recall",
    y_title="Precision",
    num_x=100,
    only_mean=False,
):
    """
    Log a metric curve visualization.

    This function generates a metric curve based on input data and logs the visualization to wandb.
    The curve can represent aggregated data (mean) or individual class data, depending on the 'only_mean' flag.

    Args:
        x (np.ndarray): Data points for the x-axis with length N.
        y (np.ndarray): Corresponding data points for the y-axis with shape (C, N), where C is the number of classes.
        names (list, optional): Names of the classes corresponding to the y-axis data; length C.
        id (str, optional): Unique identifier for the logged data in wandb.
        title (str, optional): Title for the visualization plot.
        x_title (str, optional): Label for the x-axis.
        y_title (str, optional): Label for the y-axis.
        num_x (int, optional): Number of interpolated data points for visualization.
        only_mean (bool, optional): Flag to indicate if only the mean curve should be plotted.

    Notes:
        The function leverages the '_custom_table' function to generate the actual visualization.
    """
    import numpy as np

    # Create new x
    if names is None:
        names = []
    x_new = np.linspace(x[0], x[-1], num_x).round(5)

    # Create arrays for logging
    x_log = x_new.tolist()
    y_log = np.interp(x_new, x, np.mean(y, axis=0)).round(3).tolist()

    if only_mean:
        table = wb.Table(data=list(zip(x_log, y_log)), columns=[x_title, y_title])
        wb.run.log({title: wb.plot.line(table, x_title, y_title, title=title)})
    else:
        classes = ["mean"] * len(x_log)
        for i, yi in enumerate(y):
            x_log.extend(x_new)  # add new x
            y_log.extend(np.interp(x_new, x, yi))  # interpolate y to new x
            classes.extend([names[i]] * len(x_new))  # add class names
        wb.log({id: _custom_table(x_log, y_log, classes, title, x_title, y_title)}, commit=False)


def _log_plots(plots, step):
    """
    Log plots to WandB at a specific step if they haven't been logged already.

    This function checks each plot in the input dictionary against previously processed plots and logs
    new or updated plots to WandB at the specified step.

    Args:
        plots (dict): Dictionary of plots to log, where keys are plot names and values are dictionaries
            containing plot metadata including timestamps.
        step (int): The step/epoch at which to log the plots in the WandB run.

    Notes:
        The function uses a shallow copy of the plots dictionary to prevent modification during iteration.
        Plots are identified by their stem name (filename without extension).
        Each plot is logged as a WandB Image object.
    """
    for name, params in plots.copy().items():  # shallow copy to prevent plots dict changing during iteration
        timestamp = params["timestamp"]
        if _processed_plots.get(name) != timestamp:
            wb.run.log({name.stem: wb.Image(str(name))}, step=step)
            _processed_plots[name] = timestamp


def on_pretrain_routine_start(trainer):
    """Initialize and start wandb project if module is present."""
    if not wb.run:
        try:  # Known unknown crash if telemetry disabled wandb.require(\"strict\")
            wandb_id_to_resume = getattr(trainer.args, "wandb_id", None) if trainer.resume else None

            init_args = {
                "project": str(trainer.args.project).replace("/", "-") if trainer.args.project else "Ultralytics",
                "name": str(trainer.args.name).replace("/", "-"),
                "config": vars(trainer.args),
            }

            if wandb_id_to_resume:
                LOGGER.info(f"Attempting to resume W&B run with ID {wandb_id_to_resume}")
                init_args["id"] = wandb_id_to_resume
                init_args["resume"] = "allow"  # Use "allow" for robustness

            wb.init(**init_args)

            # Store the run ID (either new or resumed) back into args for checkpointing
            if wb.run:
                trainer.args.wandb_id = wb.run.id
                LOGGER.info(f"W&B run ID {wb.run.id} initialized successfully.")
            else:  # Handle case where init might fail silently or resuming failed and returned None
                LOGGER.warning("WandB initialization failed or did not return a run object.")
                trainer.args.wandb_id = None  # Ensure it's None if init fails

        except Exception as e:
            LOGGER.warning(f"WARNING ⚠️ WandB installed but not initialized correctly, not logging this run. {e}")
            # wb = None # Do not set wb to None globally


def _log_per_class_metrics(metrics, names, step):
    """Helper function to log per-class metrics based on validator type."""
    if not (metrics and names):
        return  # Do nothing if metrics or names are missing

    per_class_log = {}
    try:
        if isinstance(metrics, (SegmentMetrics, PoseMetrics)):  # Handles Segment and Pose (inherits)
            # Use box metrics' class index as reference (should be same for seg/pose)
            if (
                hasattr(metrics.box, "ap_class_index")
                and hasattr(metrics.box, "p")
                and hasattr(metrics.box, "r")
                and hasattr(metrics.box, "all_ap")
                and hasattr(metrics.box, "maps")
            ):
                metric_box = metrics.box
                metric_seg_or_pose = metrics.seg if isinstance(metrics, SegmentMetrics) else metrics.pose
                metric_suffix = "(M)" if isinstance(metrics, SegmentMetrics) else "(P)"

                if (
                    hasattr(metric_seg_or_pose, "p")
                    and hasattr(metric_seg_or_pose, "r")
                    and hasattr(metric_seg_or_pose, "all_ap")
                    and hasattr(metric_seg_or_pose, "maps")
                ):
                    num_classes_with_data = len(metric_box.ap_class_index)
                    # Basic length check
                    if (
                        len(metric_box.p) == num_classes_with_data
                        and len(metric_seg_or_pose.p) == num_classes_with_data
                    ):
                        for idx, class_index in enumerate(metric_box.ap_class_index):
                            class_name = names.get(class_index, f"class_{class_index}")
                            # Box Metrics
                            per_class_log[f"class/{class_name}/P(B)"] = metric_box.p[idx]
                            per_class_log[f"class/{class_name}/R(B)"] = metric_box.r[idx]
                            per_class_log[f"class/{class_name}/mAP50(B)"] = metric_box.all_ap[idx, 0]
                            per_class_log[f"class/{class_name}/mAP50-95(B)"] = metric_box.maps[idx]
                            # Mask/Pose Metrics
                            per_class_log[f"class/{class_name}/P{metric_suffix}"] = metric_seg_or_pose.p[idx]
                            per_class_log[f"class/{class_name}/R{metric_suffix}"] = metric_seg_or_pose.r[idx]
                            per_class_log[f"class/{class_name}/mAP50{metric_suffix}"] = metric_seg_or_pose.all_ap[
                                idx, 0
                            ]
                            per_class_log[f"class/{class_name}/mAP50-95{metric_suffix}"] = metric_seg_or_pose.maps[idx]
                    else:
                        LOGGER.warning(f"W&B Callback: Length mismatch in {type(metrics).__name__} metrics arrays.")
                else:
                    LOGGER.warning(
                        f"W&B Callback: {type(metrics).__name__} seg/pose metrics object missing attributes."
                    )
            else:
                LOGGER.warning(f"W&B Callback: {type(metrics).__name__} box metrics object missing attributes.")

        elif isinstance(metrics, (DetMetrics, OBBMetrics)):  # Handles Det and OBB (which uses metrics.box internally)
            metric_source = metrics.box  # Both DetMetrics and OBBMetrics use .box
            metric_suffix = "(B)"
            if (
                hasattr(metric_source, "ap_class_index")
                and hasattr(metric_source, "p")
                and hasattr(metric_source, "r")
                and hasattr(metric_source, "all_ap")
                and hasattr(metric_source, "maps")
            ):
                num_classes_with_data = len(metric_source.ap_class_index)
                if len(metric_source.p) == num_classes_with_data:  # Basic length check
                    for idx, class_index in enumerate(metric_source.ap_class_index):
                        class_name = names.get(class_index, f"class_{class_index}")
                        per_class_log[f"class/{class_name}/P{metric_suffix}"] = metric_source.p[idx]
                        per_class_log[f"class/{class_name}/R{metric_suffix}"] = metric_source.r[idx]
                        per_class_log[f"class/{class_name}/mAP50{metric_suffix}"] = metric_source.all_ap[idx, 0]
                        per_class_log[f"class/{class_name}/mAP50-95{metric_suffix}"] = metric_source.maps[idx]
                else:
                    LOGGER.warning(f"W&B Callback: Length mismatch in {type(metrics).__name__} metrics arrays.")
            else:
                LOGGER.warning(f"W&B Callback: {type(metrics).__name__} metrics object missing attributes.")

        else:
            LOGGER.warning(f"W&B Callback: Unrecognized metrics type {type(metrics).__name__} for per-class logging.")

        if per_class_log:
            wb.run.log(per_class_log, step=step)

    except Exception as e:
        LOGGER.error(f"W&B Callback: Error during per-class metric logging: {e}", exc_info=True)  # Log traceback


def on_fit_epoch_end(trainer):
    """Log training metrics and model information at the end of an epoch."""
    if wb.run:
        # Log standard combined metrics
        wb.run.log(trainer.metrics, step=trainer.epoch + 1)

        # Log per-class metrics using helper function
        metrics = getattr(trainer.validator, "metrics", None)
        names = getattr(trainer.validator, "names", None)
        _log_per_class_metrics(metrics, names, trainer.epoch + 1)  # Call the helper

        # Log plots and model info
        _log_plots(trainer.plots, step=trainer.epoch + 1)
        _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
        if trainer.epoch == 0:
            wb.run.log(model_info_for_loggers(trainer), step=trainer.epoch + 1)


def on_train_epoch_end(trainer):
    """Log metrics and save images at the end of each training epoch."""
    wb.run.log(trainer.label_loss_items(trainer.tloss, prefix="train"), step=trainer.epoch + 1)
    wb.run.log(trainer.lr, step=trainer.epoch + 1)
    if trainer.epoch == 1:
        _log_plots(trainer.plots, step=trainer.epoch + 1)


def on_train_end(trainer):
    """Save the best model as an artifact and log final plots at the end of training."""
    _log_plots(trainer.validator.plots, step=trainer.epoch + 1)
    _log_plots(trainer.plots, step=trainer.epoch + 1)
    art = wb.Artifact(type="model", name=f"run_{wb.run.id}_model")
    if trainer.best.exists():
        art.add_file(trainer.best)
        wb.run.log_artifact(art, aliases=["best"])
    # Check if we actually have plots to save
    if trainer.args.plots and hasattr(trainer.validator.metrics, "curves_results"):
        for curve_name, curve_values in zip(trainer.validator.metrics.curves, trainer.validator.metrics.curves_results):
            x, y, x_title, y_title = curve_values
            _plot_curve(
                x,
                y,
                names=list(trainer.validator.metrics.names.values()),
                id=f"curves/{curve_name}",
                title=curve_name,
                x_title=x_title,
                y_title=y_title,
            )
    wb.run.finish()  # required or run continues on dashboard


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if wb
    else {}
)
