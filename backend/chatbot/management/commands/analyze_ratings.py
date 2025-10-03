"""
Django management command to analyze user ratings and calculate metrics
"""
from chatbot.rating_analyzer import RatingAnalyzer
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = 'Analyze user ratings and calculate legal accuracy metrics'

    def add_arguments(self, parser):
        parser.add_argument(
            '--days',
            type=int,
            default=30,
            help='Number of days to analyze (default: 30)'
        )
        parser.add_argument(
            '--export',
            action='store_true',
            help='Export results to JSON file'
        )
        parser.add_argument(
            '--output',
            type=str,
            help='Output file name for export'
        )
        parser.add_argument(
            '--verbose',
            action='store_true',
            help='Show detailed analysis'
        )

    def handle(self, *args, **options):
        try:
            # Initialize analyzer
            analyzer = RatingAnalyzer()
            
            # Generate report
            self.stdout.write(f"Analyzing ratings for last {options['days']} days...")
            report = analyzer.generate_report(days_back=options['days'])
            
            # Check for errors
            if 'error' in report:
                raise CommandError(f"Analysis failed: {report['error']}")
            
            # Print summary
            if 'summary' in report:
                self.stdout.write(self.style.SUCCESS('\n' + report['summary']))
            else:
                self.stdout.write(self.style.ERROR('No summary available'))
                return
            
            # Show detailed analysis if verbose
            if options['verbose']:
                self._show_detailed_analysis(report)
            
            # Export if requested
            if options['export']:
                output_file = analyzer.export_metrics(report, options['output'])
                self.stdout.write(
                    self.style.SUCCESS(f'Results exported to: {output_file}')
                )
            
            self.stdout.write(
                self.style.SUCCESS('Rating analysis completed successfully!')
            )
            
        except Exception as e:
            raise CommandError(f'Analysis failed: {str(e)}')

    def _show_detailed_analysis(self, report):
        """Show detailed analysis breakdown"""
        self.stdout.write('\n' + '='*50)
        self.stdout.write('DETAILED ANALYSIS')
        self.stdout.write('='*50)
        
        # Overall metrics
        overall = report.get('overall_metrics', {})
        accuracy = overall.get('accuracy', {})
        content = overall.get('content', {})
        
        self.stdout.write(f"\nOVERALL METRICS:")
        self.stdout.write(f"  Accuracy: {accuracy.get('accuracy', 0)*100:.1f}%")
        self.stdout.write(f"  F1 Score: {accuracy.get('f1_score', 0)*100:.1f}%")
        self.stdout.write(f"  Precision: {accuracy.get('precision', 0)*100:.1f}%")
        self.stdout.write(f"  Recall: {accuracy.get('recall', 0)*100:.1f}%")
        self.stdout.write(f"  Specificity: {accuracy.get('specificity', 0)*100:.1f}%")
        
        self.stdout.write(f"\nCONTENT METRICS:")
        self.stdout.write(f"  Helpfulness: {content.get('avg_helpfulness', 0):.1f}/5.0")
        self.stdout.write(f"  Clarity: {content.get('avg_clarity', 0):.1f}/5.0")
        self.stdout.write(f"  Confidence: {content.get('avg_confidence', 0):.1f}/5.0")
        
        # User analysis
        user_analysis = report.get('user_analysis', {})
        if user_analysis:
            self.stdout.write(f"\nUSER ANALYSIS ({len(user_analysis)} users):")
            for user_id, metrics in user_analysis.items():
                acc = metrics['accuracy_metrics']
                cont = metrics['content_metrics']
                self.stdout.write(f"  {user_id}: {acc.get('accuracy', 0)*100:.1f}% accuracy, "
                                f"{cont.get('avg_helpfulness', 0):.1f}/5 helpfulness "
                                f"({metrics['rating_count']} ratings)")
        
        # Case type analysis
        case_analysis = report.get('case_analysis', {})
        if case_analysis:
            self.stdout.write(f"\nCASE TYPE ANALYSIS:")
            for case_type, metrics in case_analysis.items():
                acc = metrics['accuracy_metrics']
                cont = metrics['content_metrics']
                self.stdout.write(f"  {case_type}: {acc.get('accuracy', 0)*100:.1f}% accuracy, "
                                f"{cont.get('avg_helpfulness', 0):.1f}/5 helpfulness "
                                f"({metrics['rating_count']} ratings)")
