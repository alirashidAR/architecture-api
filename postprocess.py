import numpy as np
import tensorflow as tf


class PostProcess:
    def __init__(self, model_path):
        self.color = [
            np.array([0, 195, 255]),  # living room       # #00c3ff
            np.array([0, 15, 255]),  # master room       # #000fff
            np.array([255, 155, 0]),  # kitchen           # #ff9b00
            np.array([255, 0, 245]),  # bathroom          # #ff00f5
            np.array([0, 115, 25]),  # dining room       # #007319
            np.array([255, 200, 215]),  # child room        # #ffc8d7
            np.array([185, 100, 255]),  # study room        # #b964ff
            np.array([195, 180, 255]),  # Second room       # #c3b4ff
            np.array([190, 245, 95]),  # guest room        # #bef55f
            np.array([0, 255, 45]),  # Balcony           # #00ff2d
            np.array([115, 255, 215]),  # Entrance          # #73ffd7
            np.array([180, 110, 50]),  # Storage           # #b46e32
            np.array([200, 235, 255]),  # Wall-in           # #c8ebff
            np.array([255, 255, 255]),  # External area     # #ffffff
            np.array([0, 0, 0]),  # Exterior Wall     # #000000
            np.array([255, 0, 0]),  # Front door        # #ff0000
            np.array([0, 0, 0]),  # Interior wall     # #000000
            np.array([255, 255, 0]),  # interior door     # #ffff00
        ]
        self.model = tf.keras.models.load_model(model_path)

    def process(
        self,
        input_images,
        gen_images,
        model_cycles=1,
        advanced=False,
        border_padding=2,
        verbose=0,
    ):
        input_images = np.array(input_images).reshape(-1, 256, 256, 3) / 255
        gen_images = np.array(gen_images).reshape(-1, 256, 256, 3)

        for i in range(model_cycles):
            gen_images = self.model.predict(gen_images, verbose=verbose)

        results = []

        for index in range(len(input_images)):
            label_list = self.get_label_list(index, gen_images)

            result = np.ones((256, 256, 3), dtype=int) * 255
            outside_area = (
                (input_images[index][:, :, 0] == 1).astype(int).reshape(256, 256, 1)
            )
            result = result * outside_area
            outside_walls = (
                (input_images[index][:, :, 0] * 255 != 0)
                + (input_images[index][:, :, 1] * 255 != 0)
                + (input_images[index][:, :, 2] * 255 != 0)
            ).reshape(256, 256, 1)
            result = result * outside_walls
            not_front_door_ = (
                (input_images[index][:, :, 0] * 255 != 255)
                + (input_images[index][:, :, 1] * 255 != 0)
                + (input_images[index][:, :, 2] * 255 != 0)
            ).reshape(256, 256, 1)
            front_door_ = (
                (input_images[index][:, :, 0] * 255 >= 240)
                * (input_images[index][:, :, 1] * 255 == 0)
                * (input_images[index][:, :, 2] * 255 == 0)
            ).reshape(256, 256, 1)
            result = result * not_front_door_ + front_door_ * (
                self.color[15].reshape(1, 1, 3)
            )

            filter_pad = np.ones((7, 7))

            for i, label_items in enumerate(label_list):
                if i >= 13:
                    break
                else:
                    image = self.post_process_image(
                        label_items, advanced, border_padding
                    )
                    image = tf.nn.conv2d(
                        image.reshape(-1, 256, 256, 1),
                        filter_pad.reshape(7, 7, 1, 1),
                        strides=(1, 1),
                        padding="SAME",
                    )
                    image = (image[0].numpy() > 0).astype(int)
                    result = result * (image != 1)
                    image = image * (self.color[i].reshape(1, 1, 3))
                    result += image

            results.append(result.astype(int))

        return results

    def res(self, value):
        return 1 if value >= 4 else 0

    def filter(self, value):
        return value if value != 0 else 255

    def filter_reverse(self, value):
        return value if value != 255 else 0

    def get_label_list(self, index, img):
        label_list = []

        mask1 = (img[index][:, :, 0] * 255 <= 70).astype(int)
        mask2 = (
            ((img[index][:, :, 1] * 255 <= 240) * (img[index][:, :, 1] * 255 >= 150))
            .astype(bool)
            .astype(int)
        )
        mask3 = (img[index][:, :, 2] * 255 >= 210).astype(int)
        living_room = mask1 * mask2 * mask3
        label_list.append(living_room)

        mask1 = (img[index][:, :, 0] * 255 <= 70).astype(int)
        mask2 = (img[index][:, :, 1] * 255 <= 70).astype(bool).astype(int)
        mask3 = (img[index][:, :, 2] * 255 >= 210).astype(int)
        master_room = mask1 * mask2 * mask3
        label_list.append(master_room)

        mask1 = (img[index][:, :, 0] * 255 >= 210).astype(int)
        mask2 = (
            ((img[index][:, :, 1] * 255 <= 185) * (img[index][:, :, 1] * 255 >= 125))
            .astype(bool)
            .astype(int)
        )
        mask3 = (img[index][:, :, 2] * 255 <= 70).astype(int)
        kitchen = mask1 * mask2 * mask3
        label_list.append(kitchen)

        mask1 = (img[index][:, :, 0] * 255 >= 210).astype(int)
        mask2 = (img[index][:, :, 1] * 255 <= 70).astype(bool).astype(int)
        mask3 = (img[index][:, :, 2] * 255 >= 200).astype(int)
        bath_room = mask1 * mask2 * mask3
        label_list.append(bath_room)

        mask1 = (img[index][:, :, 0] * 255 <= 70).astype(int)
        mask2 = (
            ((img[index][:, :, 1] * 255 <= 135) * (img[index][:, :, 1] * 255 >= 85))
            .astype(bool)
            .astype(int)
        )
        mask3 = (img[index][:, :, 2] * 255 <= 70).astype(int)
        dining_room = mask1 * mask2 * mask3
        label_list.append(dining_room)

        mask1 = (img[index][:, :, 0] * 255 >= 210).astype(int)
        mask2 = (
            ((img[index][:, :, 1] * 255 <= 230) * (img[index][:, :, 1] * 255 >= 170))
            .astype(bool)
            .astype(int)
        )
        mask3 = (
            (img[index][:, :, 2] * 255 >= 185) * (img[index][:, :, 2] * 255 <= 245)
        ).astype(int)
        child_room = mask1 * mask2 * mask3
        label_list.append(child_room)

        mask1 = (
            (img[index][:, :, 0] * 255 <= 210) * (img[index][:, :, 0] * 255 >= 155)
        ).astype(int)
        mask2 = (
            ((img[index][:, :, 1] * 255 <= 130) * (img[index][:, :, 1] * 255 >= 70))
            .astype(bool)
            .astype(int)
        )
        mask3 = (img[index][:, :, 2] * 255 >= 210).astype(int)
        study_room = mask1 * mask2 * mask3
        label_list.append(study_room)

        mask1 = (
            (img[index][:, :, 0] * 255 <= 220) * (img[index][:, :, 0] * 255 >= 165)
        ).astype(int)
        mask2 = (
            ((img[index][:, :, 1] * 255 <= 200) * (img[index][:, :, 1] * 255 >= 160))
            .astype(bool)
            .astype(int)
        )
        mask3 = (img[index][:, :, 2] * 255 >= 210).astype(int)
        second_room = mask1 * mask2 * mask3
        label_list.append(second_room)

        mask1 = (
            (img[index][:, :, 0] * 255 <= 220) * (img[index][:, :, 0] * 255 >= 160)
        ).astype(int)
        mask2 = (img[index][:, :, 1] * 255 >= 230).astype(bool).astype(int)
        mask3 = (
            (img[index][:, :, 2] * 255 >= 65) * (img[index][:, :, 2] * 255 >= 115)
        ).astype(int)
        guest_room = mask1 * mask2 * mask3
        label_list.append(guest_room)

        mask1 = (img[index][:, :, 0] * 255 <= 70).astype(int)
        mask2 = (img[index][:, :, 1] * 255 >= 240).astype(bool).astype(int)
        mask3 = (
            (img[index][:, :, 2] * 255 <= 65) * (img[index][:, :, 2] * 255 >= 15)
        ).astype(int)
        balcony = mask1 * mask2 * mask3
        label_list.append(balcony)

        mask1 = (
            (img[index][:, :, 0] * 255 <= 145) * (img[index][:, :, 0] * 255 >= 85)
        ).astype(int)
        mask2 = (img[index][:, :, 1] * 255 >= 240).astype(bool).astype(int)
        mask3 = (
            (img[index][:, :, 2] * 255 >= 190) * (img[index][:, :, 2] * 255 <= 230)
        ).astype(int)
        entrance = mask1 * mask2 * mask3
        label_list.append(entrance)

        mask1 = (
            (img[index][:, :, 0] * 255 <= 205) * (img[index][:, :, 0] * 255 >= 155)
        ).astype(int)
        mask2 = (
            ((img[index][:, :, 1] * 255 <= 140) * (img[index][:, :, 1] * 255 >= 80))
            .astype(bool)
            .astype(int)
        )
        mask3 = (img[index][:, :, 2] * 255 <= 70).astype(int)
        storage = mask1 * mask2 * mask3
        label_list.append(storage)

        mask1 = (
            (img[index][:, :, 0] * 255 <= 240) * (img[index][:, :, 0] * 255 >= 170)
        ).astype(int)
        mask2 = (img[index][:, :, 1] * 255 >= 205).astype(bool).astype(int)
        mask3 = (img[index][:, :, 2] * 255 >= 240).astype(int)
        wall_in = mask1 * mask2 * mask3
        label_list.append(wall_in)

        mask1 = (img[index][:, :, 0] * 255 >= 240).astype(int)
        mask2 = (img[index][:, :, 1] * 255 >= 240).astype(bool).astype(int)
        mask3 = (img[index][:, :, 2] * 255 >= 240).astype(int)
        external_area = mask1 * mask2 * mask3
        label_list.append(external_area)

        mask1 = (img[index][:, :, 0] * 255 <= 70).astype(int)
        mask2 = (img[index][:, :, 1] * 255 <= 70).astype(bool).astype(int)
        mask3 = (img[index][:, :, 2] * 255 <= 70).astype(int)
        exterior_wall = mask1 * mask2 * mask3
        label_list.append(exterior_wall)

        mask1 = (img[index][:, :, 0] * 255 >= 240).astype(int)
        mask2 = (img[index][:, :, 1] * 255 <= 70).astype(bool).astype(int)
        mask3 = (img[index][:, :, 2] * 255 <= 70).astype(int)
        front_door = mask1 * mask2 * mask3
        label_list.append(front_door)

        mask1 = (img[index][:, :, 0] * 255 <= 70).astype(int)
        mask2 = (img[index][:, :, 1] * 255 <= 70).astype(bool).astype(int)
        mask3 = (img[index][:, :, 2] * 255 <= 70).astype(int)
        interior_wall = mask1 * mask2 * mask3
        label_list.append(interior_wall)

        mask1 = (img[index][:, :, 0] * 255 >= 240).astype(int)
        mask2 = (img[index][:, :, 1] * 255 >= 240).astype(bool).astype(int)
        mask3 = (img[index][:, :, 2] * 255 <= 70).astype(int)
        interior_door = mask1 * mask2 * mask3
        label_list.append(interior_door)

        return label_list

    def get_vectors(self, image):
        val = image
        rev_val_i = np.flip(val, axis=1)
        rev_val_j = np.flip(val, axis=0)

        larmin = np.argmax(val, axis=1)
        func = np.vectorize(self.filter)
        larmin = func(larmin)

        rarmin = 255 - np.argmax(rev_val_i, axis=1)
        func = np.vectorize(self.filter_reverse)
        rarmin = func(rarmin)

        tarmin = np.argmax(val, axis=0)
        func = np.vectorize(self.filter)
        tarmin = func(tarmin)

        barmin = 255 - np.argmax(rev_val_j, axis=0)
        func = np.vectorize(self.filter_reverse)
        barmin = func(barmin)

        l = 0
        for i in range(len(larmin)):
            if larmin[i] == 255:
                continue
            else:
                if tf.math.abs(larmin[i] - l) >= 13:
                    l = larmin[i + 5]
                larmin[i] = l

        r = 255
        for i in range(len(rarmin)):
            if rarmin[i] == 0:
                continue
            else:
                if tf.math.abs(rarmin[i] - r) >= 13:
                    r = rarmin[i + 5]
                rarmin[i] = r

        t = 0
        for i in range(len(tarmin)):
            if tarmin[i] == 255:
                continue
            else:
                if tf.math.abs(tarmin[i] - t) >= 13:
                    t = tarmin[i + 5]
                tarmin[i] = t

        b = 255
        for i in range(len(barmin)):
            if barmin[i] == 0:
                continue
            else:
                if tf.math.abs(barmin[i] - b) >= 13:
                    b = barmin[i + 5]
                barmin[i] = b

        return larmin, rarmin, tarmin, barmin

    def post_process_image(self, black, advanced=True, border_padding_takeaway=0):
        filter_lr = np.array(
            [
                [1, 0, 1],
            ]
        ).reshape(1, 3, 1)

        filter_td = np.array(
            [
                [1],
                [0],
                [1],
            ]
        ).reshape(3, 1, 1)

        filter_s = np.array(
            [
                [-1, 1, 1, 1, -1],
                [0, 0, 0, 0, 0],
                [-1, 1, 1, 1, -1],
            ]
        ).reshape(3, 5, 1)

        filter_st = np.array(
            [
                [-1, 0, -1],
                [1, 0, 1],
                [1, 0, 1],
                [1, 0, 1],
                [-1, 0, -1],
            ]
        ).reshape(5, 3, 1)

        filter_ft = np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ]
        ).reshape(3, 3, 1)

        filter_p = np.array(
            [
                [-1, -1, -1],
                [0, 1, 0],
                [-1, -1, -1],
            ]
        ).reshape(3, 3, 1)

        filter_pt = np.array(
            [
                [-1, 0, -1],
                [-1, 1, -1],
                [-1, 0, -1],
            ]
        ).reshape(3, 3, 1)

        filter_c = np.array(
            [
                [-1, 1, -1],
                [1, 1, 1],
                [-1, 1, -1],
            ]
        ).reshape(3, 3, 1)

        new_img = black.reshape(-1, 256, 256, 1)

        if advanced:
            new_img = tf.nn.conv2d(
                new_img.reshape(-1, 256, 256, 1),
                filter_lr.reshape(1, 3, 1, 1),
                strides=(1, 1),
                padding="SAME",
            )
            new_img = (new_img.numpy() > 0).astype(int)
            new_img = tf.nn.conv2d(
                new_img.reshape(-1, 256, 256, 1),
                filter_td.reshape(3, 1, 1, 1),
                strides=(1, 1),
                padding="SAME",
            )
            new_img = (new_img.numpy() > 0).astype(int)
            new_img = tf.nn.conv2d(
                new_img.reshape(-1, 256, 256, 1),
                filter_s.reshape(3, 5, 1, 1),
                strides=(1, 1),
                padding="SAME",
            )
            new_img = (new_img.numpy() > 0).astype(int)
            new_img = tf.nn.conv2d(
                new_img.reshape(-1, 256, 256, 1),
                filter_st.reshape(5, 3, 1, 1),
                strides=(1, 1),
                padding="SAME",
            )
            new_img = (new_img.numpy() > 0).astype(int)
            new_img = tf.nn.conv2d(
                new_img.reshape(-1, 256, 256, 1),
                filter_ft.reshape(3, 3, 1, 1),
                strides=(1, 1),
                padding="SAME",
            )
            new_img = (
                (
                    (new_img.numpy() == 4)
                    + (new_img.numpy() == 3)
                    + (new_img.numpy() == 2)
                )
                > 0
            ).astype(int)
            semi_bench = new_img

            new_img = tf.nn.conv2d(
                new_img.reshape(-1, 256, 256, 1),
                filter_lr.reshape(1, 3, 1, 1),
                strides=(1, 1),
                padding="SAME",
            )
            new_img = (new_img.numpy() > 0).astype(int)
            new_img = tf.nn.conv2d(
                new_img.reshape(-1, 256, 256, 1),
                filter_td.reshape(3, 1, 1, 1),
                strides=(1, 1),
                padding="SAME",
            )
            new_img = (new_img.numpy() > 0).astype(int)
            new_img = tf.nn.conv2d(
                new_img.reshape(-1, 256, 256, 1),
                filter_s.reshape(3, 5, 1, 1),
                strides=(1, 1),
                padding="SAME",
            )
            new_img = (new_img.numpy() > 0).astype(int)
            new_img = tf.nn.conv2d(
                new_img.reshape(-1, 256, 256, 1),
                filter_st.reshape(5, 3, 1, 1),
                strides=(1, 1),
                padding="SAME",
            )
            new_img = (new_img.numpy() > 0).astype(int)
            new_img = tf.nn.conv2d(
                new_img.reshape(-1, 256, 256, 1),
                filter_ft.reshape(3, 3, 1, 1),
                strides=(1, 1),
                padding="SAME",
            )
            new_img = (
                (
                    (new_img.numpy() == 4)
                    + (new_img.numpy() == 3)
                    + (new_img.numpy() == 2)
                )
                > 0
            ).astype(int)

            new_img = tf.nn.conv2d(
                new_img.reshape(-1, 256, 256, 1),
                filter_lr.reshape(1, 3, 1, 1),
                strides=(1, 1),
                padding="SAME",
            )
            new_img = (new_img.numpy() > 0).astype(int)
            new_img = tf.nn.conv2d(
                new_img.reshape(-1, 256, 256, 1),
                filter_td.reshape(3, 1, 1, 1),
                strides=(1, 1),
                padding="SAME",
            )
            new_img = (new_img.numpy() > 0).astype(int)
            new_img = tf.nn.conv2d(
                new_img.reshape(-1, 256, 256, 1),
                filter_s.reshape(3, 5, 1, 1),
                strides=(1, 1),
                padding="SAME",
            )
            new_img = (new_img.numpy() > 0).astype(int)
            new_img = tf.nn.conv2d(
                new_img.reshape(-1, 256, 256, 1),
                filter_st.reshape(5, 3, 1, 1),
                strides=(1, 1),
                padding="SAME",
            )
            new_img = (new_img.numpy() > 0).astype(int)
            new_img = tf.nn.conv2d(
                new_img.reshape(-1, 256, 256, 1),
                filter_ft.reshape(3, 3, 1, 1),
                strides=(1, 1),
                padding="SAME",
            )
            new_img = (
                (
                    (new_img.numpy() == 4)
                    + (new_img.numpy() == 3)
                    + (new_img.numpy() == 2)
                )
                > 0
            ).astype(int)

            bench = new_img

            # plt.imshow((new_img[0])*255,cmap='gray')

            new_img = bench

            for i in range(10):
                new_img = tf.nn.conv2d(
                    new_img.reshape(-1, 256, 256, 1),
                    filter_p.reshape(3, 3, 1, 1),
                    strides=(1, 1),
                    padding="SAME",
                )
                new_img = (new_img.numpy() == -5).astype(int)
                new_img = tf.nn.conv2d(
                    new_img.reshape(-1, 256, 256, 1),
                    filter_pt.reshape(3, 3, 1, 1),
                    strides=(1, 1),
                    padding="SAME",
                )
                new_img = (new_img.numpy() == -5).astype(int)
                new_img = tf.nn.conv2d(
                    new_img.reshape(-1, 256, 256, 1),
                    filter_ft.reshape(3, 3, 1, 1),
                    strides=(1, 1),
                    padding="SAME",
                )
                new_img = (new_img.numpy() == 4).astype(int)

                new_img = tf.nn.conv2d(
                    new_img.reshape(-1, 256, 256, 1),
                    filter_p.reshape(3, 3, 1, 1) * -1,
                    strides=(1, 1),
                    padding="SAME",
                )
                new_img = (new_img.numpy() > 0).astype(int)
                new_img = tf.nn.conv2d(
                    new_img.reshape(-1, 256, 256, 1),
                    filter_pt.reshape(3, 3, 1, 1) * -1,
                    strides=(1, 1),
                    padding="SAME",
                )
                new_img = (new_img.numpy() > 0).astype(int)
                new_img = tf.nn.conv2d(
                    new_img.reshape(-1, 256, 256, 1),
                    filter_ft.reshape(3, 3, 1, 1),
                    strides=(1, 1),
                    padding="SAME",
                )
                new_img = (
                    (
                        (new_img.numpy() == 4)
                        + (new_img.numpy() == 3)
                        + (new_img.numpy() == 2)
                    )
                    > 0
                ).astype(int)

                new_img = tf.nn.conv2d(
                    new_img.reshape(-1, 256, 256, 1),
                    filter_c.reshape(3, 3, 1, 1) * -1,
                    strides=(1, 1),
                    padding="SAME",
                )
                new_img = (new_img.numpy() < 0).astype(int)
                new_img = tf.nn.conv2d(
                    new_img.reshape(-1, 256, 256, 1),
                    filter_c.reshape(3, 3, 1, 1) * -1,
                    strides=(1, 1),
                    padding="SAME",
                )
                new_img = (new_img.numpy() < 0).astype(int)

        # plt.imshow(bench[0]*255,cmap="gray")

        larmin, rarmin, tarmin, barmin = self.get_vectors(new_img[0])

        if not advanced:
            larmin, rarmin, tarmin, barmin = (
                larmin + border_padding_takeaway,
                rarmin - border_padding_takeaway,
                tarmin + border_padding_takeaway,
                barmin - border_padding_takeaway,
            )

        i_ = np.fromfunction(lambda i, j: i, (256, 256))
        j_ = np.fromfunction(lambda i, j: j, (256, 256))

        mask1 = (j_ > larmin).astype(int)
        mask2 = (j_ < rarmin).astype(int)
        mask3 = (i_ > tarmin.T).astype(int)
        mask4 = (i_ < barmin.T).astype(int)

        result = mask1 + mask2 + mask3 + mask4
        func = np.vectorize(self.res)
        result = func(result)
        result = result.reshape(256, 256, 1)

        return result
