"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navigation = [
  {
    title: "OVERVIEW",
    items: [
      {
        label: "Command Center",
        href: "/",
        icon: "⌁",
      },
    ],
  },
  {
    title: "INVESTIGATION",
    items: [
      {
        label: "Graph Explorer",
        href: "/graph",
        icon: "⌘",
      },
      {
        label: "Model Laboratory",
        href: "/models",
        icon: "◈",
      },
      {
        label: "Transaction Analysis",
        href: "/transactions",
        icon: "◎",
      },
      {
        label: "Feature Intelligence",
        href: "/feature-intelligence",
        icon: "◇",
      },
    ],
  },
  {
    title: "RESEARCH",
    items: [
      {
        label: "Research Analytics",
        href: "/analytics",
        icon: "▥",
      },
      {
        label: "Experiments",
        href: "/experiments",
        icon: "↗",
      },
      {
        label: "Error Analysis",
        href: "/error-analysis",
        icon: "⌁",
      },
    ],
  },
];

export default function Navbar() {
  const pathname = usePathname();

  return (
    <aside className="gs-sidebar">

      {/* =====================================================
          BRAND
          ===================================================== */}

      <div className="gs-brand">

        <div className="gs-brand-mark">
          G
        </div>

        <div className="gs-brand-text">
          <div className="gs-brand-name">
            GRAPHSHIELD
          </div>

          <div className="gs-brand-subtitle">
            FRAUD INTELLIGENCE
            <br />
            PLATFORM
          </div>
        </div>

      </div>


      {/* =====================================================
          NAVIGATION
          ===================================================== */}

      <nav className="gs-sidebar-nav">

        {navigation.map((section) => (

          <div
            className="gs-nav-section"
            key={section.title}
          >

            <div className="gs-nav-section-title">
              {section.title}
            </div>


            <div className="gs-nav-items">

              {section.items.map((item) => {

                const isActive =
                  item.href === "/"
                    ? pathname === "/"
                    : pathname === item.href ||
                      pathname.startsWith(
                        `${item.href}/`
                      );

                return (

                  <Link
                    href={item.href}
                    key={item.href}
                    className={`gs-sidebar-link ${
                      isActive
                        ? "active"
                        : ""
                    }`}
                  >

                    <span className="gs-sidebar-icon">
                      {item.icon}
                    </span>

                    <span>
                      {item.label}
                    </span>

                  </Link>

                );
              })}

            </div>

          </div>

        ))}

      </nav>


      {/* =====================================================
          SIDEBAR FOOTER
          ===================================================== */}

      <div className="gs-sidebar-footer">

        <div className="gs-research-status">

          <span className="gs-status-dot" />

          <div>

            <strong>
              Research Environment
            </strong>

            <span>
              All experiments loaded
            </span>

          </div>

        </div>


        <div className="gs-sidebar-version">
          GRAPHSHIELD v1.0 · 2026
        </div>

      </div>

    </aside>
  );
}