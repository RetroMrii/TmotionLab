const paths = {
  arrow: "M5 12h14m-6-6 6 6-6 6",
  plus: "M5 12h14M12 5v14",
  minus: "M5 12h14",
  menu: "M4 6h16M4 12h16M4 18h16",
  close: "m6 6 12 12M18 6 6 18",
};

export function InterfaceIcon({ name, className = "" }: {
  name: keyof typeof paths;
  className?: string;
}) {
  return <svg className={`interface-icon ${className}`} viewBox="0 0 24 24"
    width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.5"
    strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
    <path d={paths[name]} />
  </svg>;
}
