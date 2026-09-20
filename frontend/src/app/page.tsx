import Navigation from "@/components/Navigation";
import Hero from "@/components/Hero";
import ProblemStatement from "@/components/ProblemStatement";
import PipelineExplorer from "@/components/PipelineExplorer";
import MetricsPanel from "@/components/MetricsPanel";
import ThresholdExplorer from "@/components/ThresholdExplorer";
import TechStack from "@/components/TechStack";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <main className="min-h-screen flex flex-col bg-canvas text-ink selection:bg-signal/20 selection:text-ink">
      <Navigation />
      <Hero />
      <ProblemStatement />
      <PipelineExplorer />
      <MetricsPanel />
      <ThresholdExplorer />
      <TechStack />
      <Footer />
    </main>
  );
}
