import dynamic from "next/dynamic";

const GeopoliticalMap = dynamic(
  () => import("@/components/GeopoliticalMap"),
  { ssr: false }
);

export default function HomePage() {
  return <GeopoliticalMap />;
}
