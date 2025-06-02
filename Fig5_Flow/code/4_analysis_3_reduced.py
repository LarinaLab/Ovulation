# This version deletes some functionality from version 3.

import pickle
import os
import subprocess

import numpy as np
import numpy.typing as npt
import nrrd
from scipy import ndimage
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt, colors as mcolors
from PIL import Image, ImageDraw, ImageFont
import aggdraw
import svgwrite


LOW_HISTOGRAM_THRESHOLD: float = 1e-2 # 0.3
HIGH_HISTOGRAM_THRESHOLD: float = 1e-6 # 1e-4
def range_from_histogram(histogram):
    """Compute the dynamic range for a linear histogram.

    The dynamic range is chosen to have a predefined fraction of saturated
    pixels on the upper and lower ends, assuming low values are transparent.
    Colors are updated if this channel used to be triangular.
    """

    s: int = histogram.size
    # Skip the first bin since histograms often contain many zeros
    # which shouldn't be counted as part of the valid image.
    n: int = histogram[1:].sum()
    i: int = 0
    sum_low: int = 0
    for i in range(1, s):
        sum_low += histogram[i]
        if sum_low > n*LOW_HISTOGRAM_THRESHOLD:
            break
    range0 = 255 * i / (s - 1)

    sum_high: int = 0
    for i in reversed(range(s)):
        sum_high += histogram[i]
        if sum_high > n*HIGH_HISTOGRAM_THRESHOLD:
            break
    range1 = 255 * i / (s - 1)

    return range0, range1


def run_ffmpeg(dir_frames, title, crf=23):
    # crf is the compression level--lower is higher quality.
    ffmpeg_command = [
        "ffmpeg",
        "-r", str(fps),
        "-i", dir_frames + "%04d.png",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-vf", "format=yuv420p",
        f"videos/{title}.mp4"
    ]
    while True:
        proc = subprocess.run(ffmpeg_command, capture_output=True)
        if proc.stdout:
            print(proc.stdout.decode())
        if proc.stderr:
            print(proc.stderr.decode())
        if proc.returncode == 0:
            return
        input("Encoding failed! Hit enter to try again.\n")


class Scalebar:
    def __init__(self, physical, scale_im, align_left=True):
        if align_left:
            self.color = (255, 255, 255)
        else:
            self.color = (255, 0, 0)
        self.fill = aggdraw.Brush((*self.color, 255))
        self.svg_fill = svgwrite.rgb(*self.color)
        self.width = round(physical / scale_im) # microns->pixels for the scalebar.
        self.height = 4  # Scalebar height (px)
        self.pad = 8  # Padding around scalebar (px)
        self.align_left = align_left

    def get_rect(self, img):
        if self.align_left:
            return (self.pad, img.height - self.pad - self.height,
                    self.pad + self.width, img.height - self.pad)
        else:
            return (img.width - self.pad - self.width,
                    img.height - self.pad - self.height,
                    img.width - self.pad,
                    img.height - self.pad)

    def draw(self, img, draw):
        rect = self.get_rect(img)
        draw.rectangle(rect, None, self.fill)
        draw.flush()

    def svg(self, img, dwg):
        rect = self.get_rect(img)
        size = (rect[2] - rect[0], rect[3] - rect[1])
        dwg.add(dwg.rect(insert=rect[:2], size=size, fill=self.svg_fill))


class VecDraw:
    COLOR = (255, 0, 0)
    SVG_STROKE = svgwrite.rgb(*COLOR)
    STROKE_WIDTH = 1.5  # px
    HEAD_LEN = 4.  # px
    ALPHA = 128  # of 255
    THETA = np.pi * (1 - 1/6)  # Angle to rotate the head lines.
    ROT_CCW = np.array([
        [np.cos(THETA), -np.sin(THETA)], 
        [np.sin(THETA),  np.cos(THETA)]])
    ROT_CW = np.array([
        [np.cos(THETA), np.sin(THETA)], 
        [-np.sin(THETA),  np.cos(THETA)]])

    @staticmethod
    def get_lines(origin, vector):
        p0 = origin[::-1]
        p1 = p0 + vector[::-1]
        magnitude = np.sqrt(np.dot(vector, vector))
        if magnitude <= 0:
            magnitude = 1
        n = VecDraw.HEAD_LEN / magnitude * vector[::-1]
        p2 = p1 + VecDraw.ROT_CCW @ n.T
        p3 = p1 + VecDraw.ROT_CW @ n.T
        # Line and head.
        return [(*p0, *p1), (*p2, *p1, *p3)]
    
    @staticmethod
    def draw(draw, origin, vector, color):
        pen = aggdraw.Pen((*VecDraw.COLOR, VecDraw.ALPHA), VecDraw.STROKE_WIDTH)
        lines = VecDraw.get_lines(origin, vector)
        for line in lines:
            draw.line(line, pen)

    @staticmethod
    def svg(dwg, origin, vector, color):
        tail, head = VecDraw.get_lines(origin, vector)
        dwg.add(dwg.line(tail[:2], tail[2:],
                         stroke=VecDraw.SVG_STROKE,
                         stroke_opacity=VecDraw.ALPHA/255,
                         stroke_width=VecDraw.STROKE_WIDTH))
        dwg.add(dwg.polyline([head[:2], head[2:4], head[4:]],
                             stroke=VecDraw.SVG_STROKE,
                             stroke_opacity=VecDraw.ALPHA/255,
                             stroke_width=VecDraw.STROKE_WIDTH,
                             fill="none"))


def savefig(name, dpi=600):
    plt.savefig(f"fig/{name}.png", dpi=dpi, bbox_inches="tight")
    plt.savefig(f"fig/{name}.svg", dpi=dpi, bbox_inches="tight")


def prepare_shared_figure():
    plt.clf()
    fig, axes = plt.subplots(1, 2, sharey=True)
    fig.set_size_inches(10, 5, forward=True)
    plt.subplots_adjust(wspace=0.05)
    return axes


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


labels = [
    "Vivo_Control_hCG14h_040824",
    "Vivo_Control_hCG14h_041524",
    "Vivo_Control_hCG14h_041724",
    "Vivo_Padrin_hCG14h_041024",
    "Vivo_Padrin_hCG14h_041524",
    "Vivo_Padrin_hCG14h_041724",
]

make = {
    "reference_figs": False,
    "summary_figs": True,
    "special_figs": False,
    "videos": False,
}

mkdir("fig")
mkdir("frames")
mkdir("videos")
plt.rcParams["font.family"] = ["Arial"]
plt.rcParams["font.size"] = 12.0  # Default = 10.0
plt.rcParams["svg.fonttype"] = "none"  # Save text directly; no paths.
font = ImageFont.truetype("arial", 24)
fps = 60.
playback_rate = 1.  # Playback at 1x real speed.
vec_scale = 5.  # Length to draw velocity vectors, (microns/sec) / px
scalebar_physical_width = 200  # microns
speedbar_physical_width = 500  # microns/sec
y_limits = [-300, 300]
undefined_vel_stdev = 200.  # microns/sec, when data is unavailable

speed_avg_fs = []
speed_se_fs = []
flow_directions = []

for la in labels:
    print(la)
    
    arr, header = nrrd.read(f"time_filtered/{la}_tf.nrrd", index_order="C")
    arr = arr[4:-4, :, :]
    scale = header["space directions"][np.eye(3) > 0]
    scale_x, scale_y, scale_t = scale  # microns, microns, ms
    scale_t /= 1000  # ms->sec
    scale_space = np.array([scale_y, scale_x])
    scale_vel = scale_space / scale_t  # microns/s
    # Factor to multiply the image width such that scale_y
    # becomes the physical scale.
    scale_im_w = scale_x / scale_y
    
    with open(f"tracks/{la}_track.pkl", "rb") as file:
        track = pickle.load(file)
    frame_labels = track["frame_labels"]
    particles = track["particles"]
    nt = len(frame_labels)

    # Scale the data into real units.
    for p in particles.values():
        p.fit_pos *= scale_space
        p.vel *= scale_vel

    vel_avg = np.zeros((nt, 2), np.float64)
    for it in range(nt):
        if len(frame_labels[it]) == 0:
            continue
        for k in frame_labels[it]:
            vel_avg[it] += particles[k].vel_at(it)
        vel_avg[it] /= len(frame_labels[it])
    pca = PCA(1)
    pca.fit_transform(vel_avg)
    eigenvec = pca.components_.reshape((2,))
    flow_directions.append(eigenvec)
    print("Direction of flow:", eigenvec[0], eigenvec[1])

    # These are technically not "speeds", but rather 1D velocities.
    speed_avg = np.empty((nt,), np.float64)
    speed_se = np.empty((nt,), np.float64)
    for it in range(nt):
        vels = np.array([particles[k].vel_at(it) for k in frame_labels[it]])
        if vels.shape[0] > 0:
            speeds = np.dot(vels, eigenvec)
            speed_avg[it] = speeds.mean()
        else:
            # Assume constant speed when indeterminate.
            speed_avg[it] = speed_avg[it-1]
        if vels.shape[0] > 1:
            speed_se[it] = np.sqrt(speeds.var() / speeds.size)
        else:
            speed_se[it] = undefined_vel_stdev

    # Make sure the dominant flow is in the positive direction.
    if speed_avg.mean() < 0:
        print("Flipping sign convention.")
        eigenvec *= -1
        speed_avg *= -1

    lpf_stdev = 60.
    speed_avg_f = ndimage.gaussian_filter1d(speed_avg, lpf_stdev)
    speed_se_f = ndimage.gaussian_filter1d(speed_se, lpf_stdev)
    speed_avg_fs.append(speed_avg_f)
    speed_se_fs.append(speed_se_f)

    if make["reference_figs"]:
        print("Making individual figures...")
        t = np.arange(nt) * scale_t
        
        # Plot X and Y.
        plt.clf()
        plt.plot(t, vel_avg[:, 1], "r-")
        plt.plot(t, vel_avg[:, 0], "g-")
        plt.ylim(y_limits)
        plt.title(la)
        plt.xlabel("Time (sec)")
        plt.ylabel("Velocity (µm/sec)")
        plt.legend(["X-velocity, avg.", "Y-velocity, avg."])
        savefig("vel_xy_" + la)

        # Plot filtered baseline velocity.
        plt.clf()
        plt.plot(t, speed_avg_f, color="red")
        plt.fill_between(
            t,
            speed_avg_f - 2*speed_se_f,
            speed_avg_f + 2*speed_se_f, color="pink")
        plt.ylim(y_limits)
        plt.title(la)
        plt.xlabel("Time (sec)")
        plt.ylabel("Velocity (µm/sec)")
        plt.legend(["Mean axial velocity (95% confidence)"])
        savefig("baseline_1d_" + la)

        # Plot unfiltered velocity.
        plt.clf()
        plt.plot(t, speed_avg, color="red")
        plt.fill_between(
            t,
            speed_avg - 2*speed_se,
            speed_avg + 2*speed_se, color="pink")
        plt.ylim(y_limits)
        plt.title(la)
        plt.xlabel("Time (sec)")
        plt.ylabel("Velocity (µm/sec)")
        plt.legend(["Mean axial velocity (95% confidence)"])
        savefig("1d_" + la)

    if make["videos"] or make["special_figs"]:
        # Make the video frames.
        # Videos must be even in width and height.
        frame_resize = (round(scale_im_w * arr.shape[2])//2*2, arr.shape[1]//2*2)
        scale_im = scale_y * arr.shape[1] / frame_resize[1]
        dir_frames_vectors = mkdir(f"frames/vecs_red_{la}/")

        histogram, _ = np.histogram(arr[::10, ::3, ::3], bins=np.arange(0, 257))
        range0, range1 = range_from_histogram(histogram)

        scalebar = Scalebar(scalebar_physical_width, scale_im)
        speedbar = Scalebar(speedbar_physical_width, vec_scale, False)
        t_step = round(playback_rate / (fps * scale_t))
        n_frames = nt // t_step
        # Save the prettiest key figure for this sequence.
        interest = abs((speed_avg_f + 0.2*speed_avg) / (speed_se_f + 0.2*speed_se))
        key_time = min(np.argmax(interest), (n_frames - 1) * t_step)
        
        print("Making video frames...")
        for fi in range(n_frames):
            if fi % 100 == 0:
                print(f"{fi} of {n_frames}")
                
            it = fi * t_step

            # Regular grey video frame.
            arr_it = 255/(range1 - range0) * (arr[it, :, :] - range0)
            np.round(arr_it, out=arr_it)
            np.clip(arr_it, 0, 255, out=arr_it)
            img_orig = Image.fromarray(arr_it.astype(np.uint8))
            if make["videos"] or fi == key_time // t_step and make["special_figs"]:
                img = img_orig.resize(frame_resize, Image.BICUBIC)
                draw = aggdraw.Draw(img)
                scalebar.draw(img, draw)

                draw = ImageDraw.Draw(img)
                sec = it * scale_t
                timestamp = f"{sec:.1f} sec"
                draw.text((6, 6), timestamp, font=font, fill=255)

                # Quiverplot video frame.
                img = img.convert("RGBA")
                draw = aggdraw.Draw(img)
                speedbar.draw(img, draw)

                heads = []
                vecs = []
                colors = []
                for k in frame_labels[it]:
                    p = particles[k]
                    colors.append(p.get_color())
                    vecs.append(p.vel_at(it) / vec_scale)
                    heads.append(p.pos_at(it) / scale_im)
                for h, v, c in zip(heads, vecs, colors):
                    VecDraw.draw(draw, h, v, c)
                draw.flush()
            if make["videos"]:
                img.save(dir_frames_vectors + f"{fi:04d}.png")

            # Save a special key frame as an SVG.
            if fi == key_time // t_step and make["special_figs"]:
                key_dir_path = mkdir(f"fig/fastest_" + la)
                key_image_path = f"fastest_{la}/image_{la}.png"
                key_particles_path = f"fastest_{la}/particles_{la}.png"
                key_mask_path = f"fastest_{la}/mask_{la}.png"
                img_orig.save("fig/" + key_image_path)
                dwg = svgwrite.Drawing(f"fig/fastest_{la}.svg")
                dwg.add(dwg.image(key_image_path, size=frame_resize,
                                  preserveAspectRatio="none"))
                dwg.add(dwg.text(timestamp, insert=(6, 12 + font.size/2),
                                 font_size=font.size,
                                 font_family=font.getname()[0],
                                 fill="white"))
                scalebar.svg(img, dwg)
                speedbar.svg(img, dwg)
                for h, v, c in zip(heads, vecs, colors):
                    VecDraw.svg(dwg, h, v, c)
                dwg.save()

        if make["videos"]:
            print("Compiling videos...")
            run_ffmpeg(dir_frames_vectors, "vecs_red_" + la, crf=8)
            

print("All flow directions:")
print(flow_directions)
print("Combined figures...")

if make["summary_figs"]:
    color_names = "orange green red purple blue gray".split()
    line_colors = [mcolors.TABLEAU_COLORS["tab:" + n] for n in color_names]
    shade_colors = np.clip(1.5 * mcolors.to_rgba_array(line_colors), 0, 1)
    shade_colors[:, 3] = 0.5
    titles = ["Vehicle", "Inhibitor"]
    
    # Plot filtered baseline velocity.
    axes = prepare_shared_figure()
    for i in range(2):
        ax = axes[i]
        line_labels = []
        for k in range(3):
            j = 3*i + k
            line_labels.append(labels[j])
            speed_avg_f = speed_avg_fs[j]
            speed_se_f = speed_se_fs[j]
            nt = speed_avg_f.size
            t = np.arange(nt) * scale_t  # Assume constant scale_t.
            ax.plot(t, speed_avg_f, color=line_colors[j])
            ax.fill_between(
                t,
                speed_avg_f - 2*speed_se_f,
                speed_avg_f + 2*speed_se_f,
                color=shade_colors[j], label="_nolegend_")
        ax.legend(line_labels, loc="upper right")
        ax.set_title(titles[i])
        ax.set_xlabel("Time (sec)")
        ax.set_xlim([0, 42])
        ax.grid(True)

    axes[0].set_ylim(y_limits)
    axes[0].set_ylabel("Velocity (µm/sec)")
    savefig("combined_legend", dpi=200)
    for i in range(2):
        axes[i].get_legend().remove()
    savefig("combined", dpi=200)

    # Violin plots
    speeds = [abs(a) for a in speed_avg_fs]
    axes = prepare_shared_figure()
    for i in range(2):
        ax = axes[i]
        parts = ax.violinplot(speeds[i*3:(i+1)*3])
        for j, body in enumerate(parts["bodies"]):
            k = i*3 + j
            body.set_edgecolor(line_colors[k])
            body.set_facecolor(shade_colors[k])
        parts["cmins"].set_color("black")
        parts["cmaxes"].set_color("black")
        parts["cbars"].set_color("black")
        ax.grid(True)
        ax.set_xticks([1, 2, 3], labels=["", "", ""])
        ax.set_xlabel(titles[i])
    axes[0].set_ylabel("Speed (µm/sec)")
    savefig("combined_violin", dpi=200)

    # Violin plots summed for each class.
    speeds = [
        np.array([abs(a) for a in speed_avg_fs[3*i:3*(i+1)]]).reshape((-1))
        for i in range(2)]
    plt.clf()
    parts = plt.violinplot(speeds)
    axis = plt.gca()
    for j, body in enumerate(parts["bodies"]):
        body.set_edgecolor(line_colors[-1])
        body.set_facecolor(shade_colors[-1])
    parts["cmins"].set_color("black")
    parts["cmaxes"].set_color("black")
    parts["cbars"].set_color("black")
    axis.grid(True)
    axis.set_xticks([1, 2], labels=titles)
    axis.set_ylabel("Speed (µm/sec)")
    savefig("combined_violin_summed", dpi=200)
