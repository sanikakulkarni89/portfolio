import SectionPage from '../components/SectionPage'

export default function Education() {
  return (
    <SectionPage color="#f5a7c7" icon="📚" title="EDUCATION">
      <p className="section-label">// academic history</p>

      <div className="timeline">
        <div className="timeline-line"></div>

        <div className="timeline-item">
          <div className="timeline-dot"></div>
          <div className="tl-date">2023 — 2025</div>
          <div className="tl-school">Syracuse University</div>
          <div className="tl-degree">Master of Science, Computer Science</div>
          <div className="tl-detail">Advanced coursework in distributed systems, machine learning, and cloud architecture. Research focus on microservices and event-driven architectures.</div>
          <span className="tl-badge">Graduated May 2025</span>
        </div>

        <div className="timeline-item">
          <div className="timeline-dot"></div>
          <div className="tl-date">2019 — 2023</div>
          <div className="tl-school">D Y Patil International University</div>
          <div className="tl-degree">Bachelor of Technology, Computer Science Engineering</div>
          <div className="tl-detail">Minor: Artificial Intelligence &amp; Machine Learning. Relevant coursework: Data Structures, Algorithms, Database Management, Machine Learning, NLP.</div>
          <span className="tl-badge">Graduated July 2023</span>
        </div>

        <div className="timeline-item">
          <div className="timeline-dot"></div>
          <div className="tl-date">Technical Skills</div>
          <div className="tl-school">Core Competencies</div>
          <div className="tl-degree">Full-Stack Development &amp; Cloud Architecture</div>
          <div className="tl-detail">Languages: C, C++, Python, Java, JavaScript, HTML, CSS, SQL. Cloud: AWS, Azure, GCP. Databases: MySQL, PostgreSQL, MongoDB, Redis. Frameworks: React, Angular, Spring Boot, Flask, Node.js.</div>
          <span className="tl-badge">PROFICIENT</span>
        </div>
      </div>
    </SectionPage>
  )
}
