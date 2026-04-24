import matplotlib.pyplot as plt
import gymnasium as gym

def visualize_env(env, save_path="env_visualization.png", title=None):
    """
    Visualize a given MiniGrid-style environment.
    Tries to use matplotlib and RGBImgObsWrapper (if available) for cleaner output, else falls back to 'human' render.
    Args:
        env: The environment instance to visualize.
        save_path: Where to save the image, if using matplotlib.
        title: Optional title for the image.
    """

    # Optionally wrap for RGB obs if available and not already wrapped
    img = None
    if gym is not None:
        try:
            from minigrid.wrappers import RGBImgObsWrapper
            # Only wrap if not already wrapped
            if not isinstance(env, RGBImgObsWrapper):
                env = RGBImgObsWrapper(env)
            obs = env.reset()
            # Compatible with gymnasium/gym output
            if isinstance(obs, tuple) and len(obs) == 2:
                obs = obs[0]
            img = obs["image"] if isinstance(obs, dict) and "image" in obs else None
        except Exception:
            img = None

    if img is not None and plt is not None:
        plt.imshow(img)
        if title is None:
            title = getattr(env, "spec", None)
            if title is not None:
                title = str(title)
            else:
                title = "Environment visualization"
        plt.title(title)
        plt.axis("off")
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Saved initial state image to {save_path}")
        plt.show()
    else:
        # Fall back to env's render window (requires 'human' mode and pygame)
        try:
            env.render("human")
        except Exception:
            print("Neither RGB image nor human rendering available for this environment.")