from manim import *

class MainScene(Scene):
    def construct(self):
        title = Title("Sine and Cosine on the Unit Circle")
        self.play(Write(title))

        circle = Circle(radius=1)
        dot = Dot(color=RED)
        radius_line = Line(ORIGIN, dot.get_center(), color=BLUE)
        angle_arc_radius = 0.4
        angle_arc = Arc(radius=angle_arc_radius, start_angle=0, angle=0, color=YELLOW)

        angle_tracker = ValueTracker(0)

        dot.move_to(circle.point_at_angle(angle_tracker.get_value()))
        radius_line.put_start_and_end_on(ORIGIN, dot.get_center())


        cos_line = Line(ORIGIN, dot.get_center()).set_color(GREEN)
        sin_line = Line(dot.get_center(), [dot.get_center()[0], 0, 0]).set_color(RED)

        cos_label = MathTex("\\cos(\\theta)", color=GREEN).next_to(cos_line, DOWN, buff=0.1)
        sin_label = MathTex("\\sin(\\theta)", color=RED).next_to(sin_line, LEFT, buff=0.1)
        theta_label = MathTex("\\theta", color=YELLOW).next_to(angle_arc, RIGHT, buff=0.1).scale(0.7)


        self.add(circle, dot, radius_line, angle_arc, cos_line, sin_line, cos_label, sin_label, theta_label)

        dot.add_updater(lambda m: m.move_to(circle.point_at_angle(angle_tracker.get_value())))
        radius_line.add_updater(lambda m: m.put_start_and_end_on(ORIGIN, dot.get_center()))
        angle_arc.add_updater(lambda m: m.become(Arc(radius=angle_arc_radius, start_angle=0, angle=angle_tracker.get_value(), color=YELLOW)))

        cos_line.add_updater(lambda m: m.put_start_and_end_on(ORIGIN, [dot.get_center()[0], 0, 0]))
        sin_line.add_updater(lambda m: m.put_start_and_end_on(dot.get_center(), [dot.get_center()[0], 0, 0]))

        cos_label.add_updater(lambda m: m.next_to(cos_line, DOWN, buff=0.1))
        sin_label.add_updater(lambda m: m.next_to(sin_line, LEFT, buff=0.1))
        theta_label.add_updater(lambda m: m.next_to(angle_arc, RIGHT, buff=0.1).scale(0.7))


        self.play(angle_tracker.animate.set_value(PI/2), run_time=2, rate_func=linear)
        self.wait(0.5)
        self.play(angle_tracker.animate.set_value(PI), run_time=2, rate_func=linear)
        self.wait(0.5)
        self.play(angle_tracker.animate.set_value(3*PI/2), run_time=2, rate_func=linear)
        self.wait(0.5)
        self.play(angle_tracker.animate.set_value(2*PI), run_time=2, rate_func=linear)
        self.wait(1)