"""Train VMAX-Net on the held-in split of the rendered footage."""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import time

import numpy as np
import torch

from . import clips as clipmod
from . import dataset as ds
from .model import VmaxNet, detection_loss
from .traindata import CropSampler

OUT = pathlib.Path("pipeline_out")


def to_torch(batch):
    image = torch.from_numpy(batch["image"]).permute(0, 3, 1, 2).float()
    image = (image - 114.0) / 58.0
    target = {k: torch.from_numpy(batch[k]) for k in
              ("heat", "mask", "index", "valid", "size", "offset", "contacts", "contact_norm")}
    return image.contiguous(memory_format=torch.channels_last), target


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=800)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--lr", type=float, default=2.5e-3)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT / "vmaxnet.pt"))
    ap.add_argument("--resume", default=None, help="checkpoint to continue from")
    ap.add_argument("--manifest", help="generated scene manifest; only train split is read")
    ap.add_argument("--labels", help="training labels file paired with the manifest")
    ap.add_argument("--max-frames", type=int, default=512)
    ap.add_argument("--frame-stride", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    if bool(args.manifest) != bool(args.labels):
        ap.error("--manifest and --labels must be supplied together")

    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    OUT.mkdir(exist_ok=True)
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    held = ", ".join(f"{k}:{v}" for k, v in clipmod.HELD_OUT_BY_SCENARIO.items())
    print(f"decoding training clips (held out -> {held})", flush=True)
    samples = (ds.build_manifest(args.manifest,args.labels,args.max_frames,args.frame_stride,args.seed)
               if args.manifest else ds.build("train", verbose=True))
    print(f"{len(samples)} labelled training frames", flush=True)
    sampler = CropSampler(samples, crop=args.crop, seed=args.seed)

    model = VmaxNet().to(device=args.device, memory_format=torch.channels_last)
    history = []
    provenance = {"dataset_kind":"legacy development clips"}
    if args.manifest:
        from .evidence import sha256
        provenance = {"manifest_sha256":sha256(args.manifest),"labels_sha256":sha256(args.labels),
                      "split":"train","sampled_frames":len(samples)}
    if args.resume:
        blob = torch.load(args.resume, map_location="cpu", weights_only=False)
        missing, unexpected = model.load_state_dict(blob["model"], strict=False)
        if missing:
            print(f"new layers initialised fresh: {', '.join(sorted({k.split('.')[0] for k in missing}))}", flush=True)
        print(f"resumed {args.resume} at step {blob.get('step')}", flush=True)
        log = OUT / "training_log.json"
        if log.exists():
            history = json.loads(log.read_text())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    offset = history[-1]["step"] if history else 0
    start = time.time()
    for step in range(1, args.iterations + 1):
        if step <= args.warmup:
            lr = args.lr * step / args.warmup
        else:
            t = (step - args.warmup) / max(args.iterations - args.warmup, 1)
            lr = args.lr * 0.5 * (1 + math.cos(math.pi * t))
        for group in opt.param_groups:
            group["lr"] = lr

        image, target = to_torch(sampler.batch(args.batch))
        image = image.to(args.device)
        target = {k:v.to(args.device) for k,v in target.items()}
        out = model(image)
        loss, parts = detection_loss(out, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 8.0)
        opt.step()

        history.append({"step": offset + step, "lr": lr, "loss": float(loss.detach()), **parts})
        if step % 10 == 0 or step == 1:
            elapsed = time.time() - start
            eta = elapsed / step * (args.iterations - step)
            print(f"{step:5d}/{args.iterations} loss {float(loss.detach()):7.3f} "
                  f"heat {parts['heat']:6.3f} contact {parts['contact']:6.4f} "
                  f"mask {parts['mask']:6.4f} lr {lr:.2e} eta {eta/60:5.1f} min", flush=True)
        if step % 100 == 0 or step == args.iterations:
            torch.save({"model": model.state_dict(), "step": offset + step,
                        "args": vars(args), "provenance": provenance}, args.out)
            (OUT / "training_log.json").write_text(json.dumps(history, indent=1))

    torch.save({"model": model.state_dict(), "step": offset + args.iterations,
                "args": vars(args), "provenance": provenance}, args.out)
    (OUT / "training_log.json").write_text(json.dumps(history, indent=1))
    print(f"saved {args.out} after {(time.time()-start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
