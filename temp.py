    def save_video(self, video_path):

        exp_vis_dir = os.path.dirname(self.cfg.logdir, "vis")
        os.makedirs(exp_vis_dir, exist_ok=True)
        video_path = os.path.join(exp_vis_dir, video_path)
        
        for i in range(self.num_envs):
            images_third = self.record_third[i]
            images_wrist = self.record_wrist[i]
            images = self.concat_images(images_third, images_wrist)


            if args.info_on_video:
                for j in range(len(infos)):
                    images[j + 1] = visualization.put_info_on_image(images[j + 1], infos[j])

            success = np.sum([d["success"] for d in infos]) >= 6
            images_to_video(images, str(exp_vis_dir), f"video_eps_{eps_count + initial_eps_count + i}_success={success}",
                            fps=30, verbose=True)